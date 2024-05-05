import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, sys
from torch.utils.tensorboard.writer import SummaryWriter
import itertools
import random
from MultiModel import MultiModel, FineTuneMode # type: ignore
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Network.Dataset import MyDataset

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

# Function to create all combinations of hyperparameters
def create_combinations(hyperparameter_grid):
    keys, values = zip(*hyperparameter_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# Function to randomly sample hyperparameters
def sample_hyperparameters(hyperparameter_grid, num_samples):
    samples = []
    for _ in range(num_samples):
        sample = {}
        for key, values in hyperparameter_grid.items():
            sample[key] = random.choice(values)
        samples.append(sample)
    return samples

def filter_hp_from_list(original_dict, state):
    new_dict = {}
    for key, value in original_dict.items():
        if isinstance(value, list):
            new_dict[key] = value[state]
        else:
            new_dict[key] = value
    return new_dict

def check_accuracy(model, dataloader, DEVICE, save_dir=None):
    model.eval()
    num_correct = 0
    num_samples = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in dataloader:
            image, label = data
            label -= 1  # Change the labels to start from 0
            label = label.type(torch.LongTensor)

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            scores = model(image)
            _, predictions = scores.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)
            
            output = (torch.max(torch.exp(scores), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            label = label.data.cpu().numpy()
            y_true.extend(label) # Save Truth

    accuracy = float(num_correct)/float(num_samples)
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy* 100:.3f}")
    print("\n\n")
    model.train()
    return accuracy

def train_net(model, loss_function, device, dataloader_train, dataloader_validation, optimizer, hyper_parameters, logger, scheduler, state, name="default"):
    epochs = hyper_parameters["epochs"]
    all_train_losses = []
    all_val_losses = []
    all_accuracies = []
    validation_loss = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')

        training_loss = 0
        model.train()  # Set the model to training mode
        train_losses = []
        accuracies = []
        for train_iteration, batch in training_loop:
            optimizer.zero_grad()  # Reset the parameter gradients for the current minibatch iteration

            images, labels = batch
            labels -= 1  # Change the labels to start from 0
            labels = labels.type(torch.LongTensor)

            labels = labels.to(device)
            images = images.to(device)

            # Forward pass, backward pass and optimizer step
            predicted_labels = model(images)
            loss_train = loss_function(predicted_labels, labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            train_losses.append(loss_train.item())

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'Train loss_{state}', loss_train.item(), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'Train accuracy_{state}', train_accuracy, epoch*len(dataloader_train)+train_iteration)
        all_train_losses.append(sum(train_losses)/len(train_losses))
        all_accuracies.append(sum(accuracies)/len(accuracies))

        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        val_losses = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                images, labels = batch
                labels -= 1  # Change the labels to start from 0
                labels = labels.type(torch.LongTensor)

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                output = model(images)

                # Calculate the loss
                loss_val = loss_function(output, labels)

                validation_loss += loss_val.item()
                val_losses.append(loss_val.item())

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'Validation loss_{state}', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        logger.add_scalars(f'Combined_{state}', {'Validation loss': validation_loss,
                                                'Train loss': training_loss/len(dataloader_train)}, epoch)
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()}")
    # logger.add_hparams(
    #     {f"Step_size_{state}": scheduler.step_size, f'Batch_size_{state}': hyper_parameters["batch size"], f'Gamma_{state}': hyper_parameters["gamma"], f'Epochs_{state}': hyper_parameters["epochs"]},
    #     {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
    #         f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
    #         f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
    # ) 

def automatic_fine_tune(logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, directory):
    STATE = 0
    new_hp = filter_hp_from_list(hyper_parameters, STATE)

    # Train new layers
    model = MultiModel(modeltype, new_hp, load_pretrained=True)
    model.fine_tune(FineTuneMode.NEW_LAYERS)
    model.to(device)    
    optimizer = optim.Adam(model.parameters(),
                       lr=new_hp["learning rate"],
                       betas=(new_hp["beta1"],
                              new_hp["beta2"]),
                       weight_decay=new_hp["weight decay"],
                       eps=new_hp["epsilon"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    train_net(model, loss_function, device, dataloader_train,
          dataloader_validation, optimizer, new_hp, logger, scheduler, state="NEW_LAYERS", name=new_hp["network name"])
    print("\nFinished training new layers!\n")

    # Train classifier layers
    if modeltype == "resnet152":
        STATE += 1
    else:
        model.fine_tune(FineTuneMode.CLASSIFIER)
        STATE += 1
        new_hp = filter_hp_from_list(hyper_parameters, STATE)
        optimizer = optim.Adam(model.parameters(),
                        lr=new_hp["learning rate"],
                        betas=(new_hp["beta1"],
                                new_hp["beta2"]),
                        weight_decay=new_hp["weight decay"],
                        eps=new_hp["epsilon"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
        train_net(model, loss_function, device, dataloader_train,
            dataloader_validation, optimizer, new_hp, logger, scheduler, state="CLASSIFIER", name=new_hp["network name"])
        print("\nFinished training classifier layers!\n")

    # Fine tune all layers
    model.fine_tune(FineTuneMode.ALL_LAYERS) # type: ignore
    STATE += 1
    new_hp = filter_hp_from_list(hyper_parameters, STATE)
    optimizer = optim.Adam(model.parameters(), # type: ignore
                       lr=new_hp["learning rate"],
                       betas=(new_hp["beta1"],
                              new_hp["beta2"]),
                       weight_decay=new_hp["weight decay"],
                       eps=new_hp["epsilon"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=new_hp["step size"], gamma=new_hp["gamma"])
    train_net(model, loss_function, device, dataloader_train,
          dataloader_validation, optimizer, new_hp, logger, scheduler, state="ALL_LAYERS", name=new_hp["network name"])
    print("\nFinished fine tuning all layers!\n")

    # Check accuracy and save model
    accuracy = check_accuracy(model, dataloader_test, device, directory)
    save_dir =  os.path.join(directory, f'accuracy_{accuracy:.3f}.pth')
    torch.save(model.state_dict(), save_dir) # type: ignore

    return accuracy

def hyperparameter_search(modeltype, loss_function, device, dataset_train, dataset_validation, dataset_test, hyperparameter_grid, missing_hp, run_dir):    
    # Initialize with a large value for minimization problems
    best_performance = 0
    best_hyperparameters = None
    run_counter = 0
    modeltype_directory = os.path.join(run_dir, f'{modeltype}')
    for hyper_parameters in hyperparameter_grid:
        # Empty memory before start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Current hyper parameters: {hyper_parameters}")
        hyper_parameters.update(missing_hp)
        # Initialize model, optimizer, scheduler, logger, dataloader
        dataloader_train = DataLoader(dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
        dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")
        dataloader_test = DataLoader(dataset_test, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for testing with batch size: {hyper_parameters['batch size']}")

        log_dir = os.path.join(modeltype_directory, f'run_{str(run_counter)}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)

        accuracy = automatic_fine_tune(logger, hyper_parameters, modeltype, device, loss_function, dataloader_train, dataloader_validation, dataloader_test, log_dir)

        run_counter += 1

        # Update best hyperparameters if the current model has better performance
        if accuracy > best_performance:
            best_performance = accuracy
            best_hyperparameters = hyper_parameters

        logger.close()
    print(f"\n\n############### Finished hyperparameter search! ###############")

    return best_hyperparameters




MEAN = np.array([0.5750, 0.6065, 0.6459])
STD = np.array([0.1854, 0.1748, 0.1794])
CLASSES = ["0", "1", "2", "3", "4"]
MODELTYPE = ["resnet152"]
# ROOT_DIRECTORY = "c:\\Users\\jacop\\Desktop\\BSc\\Code\\WindTurbineImagesCategorization\\Data\\DatasetPNG"
ROOT_DIRECTORY = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/Data/DatasetPNG"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")

transform = transformsV2.Compose([
    transformsV2.Pad(300, padding_mode="reflect"),
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),  
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    transformsV2.CenterCrop(224),
    transformsV2.Resize((224, 224)), # Adjustable
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 

run_dir = "MultiModel"
os.makedirs(run_dir, exist_ok=True)

# Define the loss function
loss_function = nn.CrossEntropyLoss()
results = {}
for model in MODELTYPE:

    hyper_parameters = {
        "network name": "",
        "input channels": 3,
        "number of classes": 5,
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "number of workers": 0,
        "beta1": 0.9, 
        "beta2": 0.999, 
        "epsilon": 1e-08,
        "weight decay": 1e-08, 
    }

    # Define your hyperparameter grid
    hyperparameter_grid = {
        'learning rate': [[0.001, 0.0001, 1e-7], [0.01, 0.00001, 1e-6]],
        'gamma': [[0.8, 0.9, 0.7], [0.1, 0.8, 0.7]],
        'batch size': [64, 128],
        "epochs": [[30, 20, 20], [30, 30, 20], [20, 35, 20]],
        'step size': [[15, 15, 20], [20, 25, 15], [10, 25, 15]],
    }

    # Create Datasets and Dataloaders
    dataset_train = MyDataset(root_directory=ROOT_DIRECTORY, mode="train", transform=transform, split=hyper_parameters["split"], pca=False)
    print(f"Created a new Dataset for training of length: {len(dataset_train)}")
    dataset_validation = MyDataset(root_directory=ROOT_DIRECTORY, mode="val", transform=None, split=hyper_parameters["split"], pca=False)
    print(f"Created a new Dataset for validation of length: {len(dataset_validation)}")
    dataset_test = MyDataset(root_directory=ROOT_DIRECTORY, mode="test", transform=None, split=hyper_parameters["split"], pca=False)
    print(f"Created a new Dataset for testing of length: {len(dataset_test)}")

    # Perform hyperparameter search
    # all_combinations = create_combinations(hyperparameter_grid)
    random_samples = sample_hyperparameters(hyperparameter_grid, 10)

    print(f"Number of combinations: {len(random_samples)} (amount of models to test)\n\n")
    best_hp = hyperparameter_search(model, loss_function, DEVICE, dataset_train, dataset_validation, dataset_test, random_samples, hyper_parameters, run_dir)
    results[model] = best_hp
    print(f"Best hyperparameters for {model}: {best_hp}")

print(f"\n\nResults: {results}")








import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from Dataset import MyDataset
from tqdm import tqdm
from Network import MyNetwork
import os
from torch.utils.tensorboard.writer import SummaryWriter
import itertools
import random


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

# def add_layer_weight_histograms(model, logger, model_name):
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
#             logger.add_histogram(f"{model_name}/{name}.weights", module.weight)

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

    model.train()
    return accuracy

def train_and_validate_net(model, loss_function, device, dataloader_train, dataloader_validation, dataloader_test, optimizer, hyper_parameters, logger, scheduler, name="default"):
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

            # Uncomment to display images in tesnorboard
            # img_grid = make_grid(images)
            # logger.add_image(f"{name}/batch images", img_grid, epoch)

            # Add the model graph to tensorboard
            # if epoch == 0 and train_iteration == 0:
            #     logger.add_graph(model, images)

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])
            accuracies.append(train_accuracy)

            # features = images.reshape(images.shape[0], -1)
            # class_labels = [CLASSES[label] for label in predicted]  # predicted

            # if epoch > 47 and train_iteration == 5:  # Only the 5th iteration of each epoch after the 27th epoch
            #     logger.add_embedding(
            #         features, metadata=class_labels, label_img=images, global_step=epoch, tag=f'{name}/Embedding')

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'{name}/Train loss', loss_train.item(),
                              epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'{name}/Train accuracy', train_accuracy,
                              epoch*len(dataloader_train)+train_iteration)
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
                logger.add_scalar(f'{name}/Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            all_val_losses.append(sum(val_losses)/len(val_losses))

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        # add_layer_weight_histograms(model, logger, name)
        logger.add_scalars(f'{name}/Combined', {'Validation loss': validation_loss,
                                                'Train loss': training_loss/len(dataloader_train)}, epoch)
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()}")
    # logger.add_hparams(
    #     {'Lr': scheduler.get_last_lr(
    #     )[0], 'Batch_size': hyper_parameters["batch size"], 'Gamma': hyper_parameters["gamma"]},
    #     {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
    #         f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
    #         f'Avg val loss': sum(all_val_losses)/len(all_val_losses)}
    # )
    accuracy = check_accuracy(model, dataloader_test, device)
    logger.add_hparams(
        {'Step_size': scheduler.step_size, 'Batch_size': hyper_parameters["batch size"], 'Gamma': hyper_parameters["gamma"], 'Lr': hyper_parameters["learning rate"], 'Epochs': hyper_parameters["epochs"]}, 
        {f'Avg train loss': sum(all_train_losses)/len(all_train_losses),
            f'Avg accuracy': sum(all_accuracies)/len(all_accuracies),
            f'Avg val loss': sum(all_val_losses)/len(all_val_losses),
            'Accuracy': accuracy}
    )
    print('-------------- Finished training a new model! --------------\n')

    return accuracy

def hyperparameter_search(loss_function, device, dataset_train, dataset_validation, dataset_test, hyperparameter_grid, missing_hp):
    # Initialize with a large value for minimization problems
    best_performance = 0
    best_hyperparameters = None
    run_counter = 0
    for hyper_parameters in hyperparameter_grid:
        # Empty memory before start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Current hyper parameters: {hyper_parameters}")
        hyper_parameters.update(missing_hp)
        # Initialize model, optimizer, scheduler, logger, dataloader
        dataloader_train = DataLoader(
            dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
        dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True,
                                           num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True,
                                    num_workers=hyper_parameters["number of workers"], drop_last=False)
        print(f"Created a new Dataloader for testing with batch size: 1")
        model = MyNetwork(hyper_parameters)
        model.weight_initialization()
        model.to(DEVICE)

        optimizer = optim.Adam(model.parameters(),
                               lr=hyper_parameters["learning rate"],
                               betas=(hyper_parameters["beta1"],
                                      hyper_parameters["beta2"]),
                               weight_decay=hyper_parameters["weight decay"],
                               eps=hyper_parameters["epsilon"])

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=hyper_parameters["step size"], gamma=hyper_parameters["gamma"])

        logger = SummaryWriter(run_dir+str(run_counter)+f" batch size_{hyper_parameters['batch size']} lr_{hyper_parameters['learning rate']} gamma_{hyper_parameters['gamma']} step_size_{hyper_parameters['step size']} epochs_{hyper_parameters['epochs']}")

        # Train the model
        accuracy = train_and_validate_net(model, loss_function, device, dataloader_train, dataloader_validation, dataloader_test,
                                      optimizer, hyper_parameters, logger, scheduler, name=hyper_parameters["network name"])

        run_counter += 1

        # Update best hyperparameters if the current model has better performance
        if accuracy < best_performance:
            best_performance = accuracy
            best_hyperparameters = hyper_parameters

        logger.close()
    print(f"\n\n############### Finished hyperparameter search! ###############")

    return best_hyperparameters


CLASSES = ["0", "1", "2", "3", "4"]
# ROOT_DIRECTORY = "c:\\Users\\jacop\\Desktop\\BSc\\Code\\WindTurbineImagesCategorization\\Data\\DatasetPCA"
ROOT_DIRECTORY = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/Data/DatasetPCA"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")
hyper_parameters = {
    "network name": "MyNetwork",\
    "PCA": True,
    "input channels": 3,
    "number of classes": 5,
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "number of workers": 0,
    "epsilon": 1e-08,
    "weight decay": 1e-08,
    'beta1': 0.9,
    'beta2': 0.999,
}

# logs_dir = 'WindTurbineImagesCategorization\\Network\\tests'
logs_dir = "HPSearch_PCA_logs"
os.makedirs(logs_dir, exist_ok=True)
run_dir = os.path.join(logs_dir, f'run_')

# Define your hyperparameter grid
hyperparameter_grid = {
    'learning rate': [0.001, 0.0001, 0.1, 0.01],
    'epochs' : [50, 20, 25, 35],
    'gamma': [0.8, 0.9, 0.7],
    'batch size': [64, 128],
    'step size': [10, 15, 20, 25],
}

# Create Datasets and Dataloaders
dataset_train = MyDataset(root_directory=ROOT_DIRECTORY, mode="train", transform=None, split=hyper_parameters["split"], pca=True)
print(f"Created a new Dataset for training of length: {len(dataset_train)}")
dataset_validation = MyDataset(root_directory=ROOT_DIRECTORY, mode="val", transform=None, split=hyper_parameters["split"], pca=True)
print(f"Created a new Dataset for validation of length: {len(dataset_validation)}\n")
dataset_test = MyDataset(root_directory=ROOT_DIRECTORY, mode="test", transform=None, split=hyper_parameters["split"], pca=True)
print(f"Created a new Dataset for testing of length: {len(dataset_test)}\n")
loss_function = nn.CrossEntropyLoss()


# Perform hyperparameter search
all_combinations = create_combinations(hyperparameter_grid)
# random_samples = sample_hyperparameters(hyperparameter_grid, 10)

print(f"Number of combinations: {len(all_combinations)} (amount of models to test)\n\n")

best_hp = hyperparameter_search(
    loss_function, DEVICE, dataset_train, dataset_validation, dataset_test, all_combinations, hyper_parameters)

print(f"Best hyperparameters: {best_hp}")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import DataLoader
from Dataset import MyDataset
from tqdm import tqdm
from Network import MyNetwork
import os
from torch.utils.tensorboard.writer import SummaryWriter
# %load_ext autoreload
# %autoreload 2
# os.environ['KMP_DUPLICATE_LIB_OK']='True' # To prevent the kernel from dying

def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)

def check_accuracy(model, dataloader, DEVICE):
    model.eval()
    num_correct = 0
    num_samples = 0
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

    accuracy = float(num_correct)/float(num_samples)
    print(f"Got {num_correct}/{num_samples} with accuracy {accuracy* 100:.3f}")
    model.train()
    return accuracy

def train_net(model, loss_function, device, dataloader_train, dataloader_validation, optimizer, hyper_parameters, logger, scheduler, name="default"):
    epochs = hyper_parameters["epochs"]

    validation_loss = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        """    Train step for one batch of data    """
        training_loop = create_tqdm_bar(
            dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')

        training_loss = 0
        model.train()  # Set the model to training mode
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

            # Running train accuracy
            _, predicted = predicted_labels.max(1)
            num_correct = (predicted == labels).sum()
            train_accuracy = float(num_correct)/float(images.shape[0])

            training_loop.set_postfix(train_loss="{:.8f}".format(
                training_loss / (train_iteration + 1)), val_loss="{:.8f}".format(validation_loss))

            logger.add_scalar(f'{name}/Train loss', loss_train.item(), epoch*len(dataloader_train)+train_iteration)
            logger.add_scalar(f'{name}/Train accuracy', train_accuracy, epoch*len(dataloader_train)+train_iteration)


        """    Validation step for one batch of data    """
        val_loop = create_tqdm_bar(
            dataloader_validation, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
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

                val_loop.set_postfix(val_loss="{:.8f}".format(
                    validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                logger.add_scalar(f'{name}/Validation loss', validation_loss/(
                    val_iteration+1), epoch*len(dataloader_validation)+val_iteration)
            

        # This value is for the progress bar of the training loop.
        validation_loss /= len(dataloader_validation)

        logger.add_scalars(f'{name}/Combined', {'Validation loss': validation_loss,
                                                'Train loss': training_loss/len(dataloader_train)}, epoch)
        scheduler.step()
        print(f"Current learning rate: {scheduler.get_last_lr()}")



MEAN = np.array([0.5750, 0.6065, 0.6459])
STD = np.array([0.1854, 0.1748, 0.1794])
# ROOT_DIRECTORY = "c:\\Users\\jacop\\Desktop\\BSc\\Code\\WindTurbineImagesCategorization\\Data\\DatasetPNG"
ROOT_DIRECTORY = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/Data/Dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")

transform = transformsV2.Compose([
    transformsV2.Resize((224, 224)),
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    # Replace deprecated ToTensor()
    transformsV2.ToImage(),
    transformsV2.ToDtype(torch.float32, scale=True),
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
])
hyper_parameters = {
    "network name": "B64",
    "input channels": 3,
    "number of classes": 5,
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "batch size": 64,
    "number of workers": 0,
    "learning rate": 0.01,
    "epochs": 500,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "weight decay": 1e-08,
    "step size": 35,
    "gamma": 0.7,
}

# Create Datasets and Dataloaders
dataset_train = MyDataset(root_directory=ROOT_DIRECTORY, mode="train", transform=transform, split=hyper_parameters["split"])
print(f"Created a new Dataset for training of length: {len(dataset_train)}")
dataset_validation = MyDataset(root_directory=ROOT_DIRECTORY, mode="val", transform=None, split=hyper_parameters["split"])
print(f"Created a new Dataset for validation of length: {len(dataset_validation)}")
dataloader_train = DataLoader(dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")
dataset_test = MyDataset(root_directory=ROOT_DIRECTORY, mode="test", transform=None, split=hyper_parameters["split"])
print(f"Created a new Dataset for testing of length: {len(dataset_test)}")
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for testing with batch size: 1")

# Create Model, initialize its weights, create optimizer and loss function
model = MyNetwork(hyper_parameters)
model.weight_initialization()
model.to(DEVICE)
print(f"Model created, move to {DEVICE} and weights initialized")
optimizer = optim.Adam(model.parameters(),
                       lr=hyper_parameters["learning rate"],
                       betas=(hyper_parameters["beta1"],
                              hyper_parameters["beta2"]),
                       weight_decay=hyper_parameters["weight decay"],
                       eps=hyper_parameters["epsilon"])
loss_function = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyper_parameters["step size"], gamma=hyper_parameters["gamma"])

# Determine the path for saving logs
logs_dir = 'run_{name}'
os.makedirs(logs_dir, exist_ok=True)
# num_of_runs = len(os.listdir(logs_dir))
# run_dir = os.path.join(logs_dir, f'run_{num_of_runs + 1}')
# logger = SummaryWriter(run_dir)
logger = SummaryWriter(logs_dir)

# Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

train_net(model, loss_function, DEVICE, dataloader_train,
          dataloader_validation, optimizer, hyper_parameters, logger, scheduler, name=hyper_parameters["network name"])

accuracy = check_accuracy(model, dataloader_test, DEVICE)

save_dir =  os.path.join("WindTurbineImagesCategorization\\Network\\Results", f'{hyper_parameters["network name"]}_accuracy_{accuracy:.3f}.pth')
torch.save(model.state_dict(), save_dir)
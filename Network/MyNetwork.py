import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import DataLoader
from Dataset import MyDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

MEAN = np.array([0.5750, 0.6065, 0.6459])
STD = np.array([0.1854, 0.1748, 0.1794])
ROOT_DIRTECTORY = "WindTurbineImagesCategorization\\Data\\Dataset"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {DEVICE}")



class MyNetwork(nn.Module):
    def __init__(self, hyper_parameters):
        super(MyNetwork, self).__init__()
        self.input_channels = hyper_parameters["input channels"]
        self.number_of_classes = hyper_parameters["number of classes"]

        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),            
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),            
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
        )
        
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(512*7*7, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.number_of_classes)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fully_connected_layers(x)
        
        return x
    
    def weight_initialization(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
    
    def count_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        
        print(f"Total number of parameters: {total_params}")









# TODO: Look up this function in the documentation
def create_tqdm_bar(iterable, desc):
    return tqdm(enumerate(iterable), total=len(iterable), ncols=150, desc=desc)


def train_net(model, loss_function, device, dataloader_train, dataloader_val, optimizer, hyper_parameters, name="default"):
    #def train_classifier(classifier, train_loader, val_loader, loss_func, tb_logger, epochs=10, name="default"):

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs * len(train_loader) / 5, gamma=0.7) Not used in this exercise
    epochs = hyper_parameters["epochs"]
    validation_loss = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        # Train
        training_loop = create_tqdm_bar(dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')
        training_loss = 0
        #for i, data in enumerate(trainloader, 0):
        for train_iteration, batch in training_loop:

            images, labels = batch
            labels -= 1

            labels = labels.type(torch.LongTensor)
            # images = images.type(torch.FloatTensor)
            
            labels = labels.to(device)
            images = images.to(device)
            
            model.train()  # Set the model to training mode
            # Reset the parameter gradients for the current minibatch iteration 
            optimizer.zero_grad()
            
            # forward + backward + optimize
            predicted_labels = model(images)

            loss_train = loss_function(predicted_labels, labels)
            loss_train.backward()
            optimizer.step()

            # Accumulate the loss and calculate the accuracy of predictions
            training_loss += loss_train.item()
            # scheduler.step()

            # _, preds = torch.max(y_pred, 1) #convert output probabilities of each class to a singular class prediction
            # correct += preds.eq(y).sum().item()
            # total += y.size(0)

            # Update the progress bar.
            training_loop.set_postfix(train_loss = "{:.8f}".format(training_loss / (train_iteration + 1)), val_loss = "{:.8f}".format(validation_loss))

            # Update the tensorboard logger.
            # tb_logger.add_scalar(f'{name}/train_loss', loss.item(), epoch * len(dataloader_train) + train_iteration)

            # print statistics
            # if train_iteration % 10 == 9:    # print every 10 batches
            #     print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch+1,
            #                                                        train_iteration+1,
            #                                                        training_loss / (len(dataloader_train)*epoch+train_iteration)))
        
        #print('Finished Training')

        # Validation
        val_loop = create_tqdm_bar(dataloader_val, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
        validation_loss = 0
        with torch.no_grad():
            for val_iteration, batch in val_loop:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    # Extract data from the batch
                    images, labels = batch
                    labels -= 1
                    labels = labels.type(torch.LongTensor)
                    
                    images = images.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    output = model(images)

                    # Calculate the loss
                    loss_val = loss_function(output, labels)

                validation_loss += loss_val.item()

                # Update the progress bar.
                val_loop.set_postfix(val_loss = "{:.8f}".format(validation_loss/(val_iteration+1)))

                # Update the tensorboard logger.
                # tb_logger.add_scalar(f'{name}/val_loss', validation_loss/(val_iteration+1), epoch*len(dataloader_val)+val_iteration)
            
            # This value is for the progress bar of the training loop.
            validation_loss /= len(dataloader_val)





transform = transformsV2.Compose([
    transformsV2.Resize((224, 224)),
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),  
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 

hyper_parameters = {
    "input channels": 3,
    "number of classes": 5,
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "batch size": 16,
    "number of workers": 0,
    "learning rate": 0.001,
    "epochs": 10,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-08,
    "weight_decay": 1e-08
}


# Create Datasets and Dataloaders
dataset_train = MyDataset(root_directory=ROOT_DIRTECTORY, mode="train", transform=transform, split=hyper_parameters["split"])
print(f"Created a new Dataset for training of length: {len(dataset_train)}")
dataset_validation = MyDataset(root_directory=ROOT_DIRTECTORY, mode="val", transform=None, split=hyper_parameters["split"])
print(f"Created a new Dataset for validation of length: {len(dataset_validation)}")

dataloader_train = DataLoader(dataset_train, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
print(f"Created a new Dataloader for training with batch size: {hyper_parameters["batch size"]}")
# dataloader_validation = DataLoader(dataset_validation, batch_size=hyper_parameters["batch size"], shuffle=True, num_workers=hyper_parameters["number of workers"], drop_last=False)
dataloader_validation = DataLoader(dataset_validation, batch_size=1, shuffle=False, num_workers=hyper_parameters["number of workers"], drop_last=False)

print(f"Created a new Dataloader for validation with batch size: {hyper_parameters["batch size"]}")

# Create Model, initialize its weights, create optimizer and loss function
model = MyNetwork(hyper_parameters)
model.weight_initialization()
model.to(DEVICE)
print(f"Model created, move to {DEVICE} and weights initialized")

optimizer = optim.Adam(model.parameters(), 
                       lr=hyper_parameters["learning rate"], 
                       betas=(hyper_parameters["beta1"], hyper_parameters["beta2"]), 
                       weight_decay=hyper_parameters["weight_decay"], 
                       eps=hyper_parameters["epsilon"])
loss_function = nn.CrossEntropyLoss()

# Empty memory before start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Starting training...")




    
train_net(model, loss_function, DEVICE, dataloader_train, dataloader_validation, optimizer, hyper_parameters, name="MyModel")


# #i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
# path = os.path.join('logs', 'cls_logs')
# num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
# path = os.path.join(path, f'run_{num_of_runs + 1}')

# tb_logger = SummaryWriter(path)


# model = KeypointModel(hparams)
# criterion = nn.MSELoss() # The loss function we use for classification.
# train_net(epochs=hparams['num_epochs'], model=model, loss_func=criterion, name="MyModel")
# train_classifier(classifier, labled_train_loader, labled_val_loader, loss_func, tb_logger, epochs=epochs, name="Default")

print("Finished training!")
#print("How did we do? Let's check the accuracy of the defaut classifier on the training and validation sets:")
# print(f"Training Acc: {classifier.getAcc(labled_train_loader)[1] * 100}%")
# print(f"Validation Acc: {classifier.getAcc(labled_val_loader)[1] * 100}%")

























# # Initializing the list for storing the loss and accuracy
# train_loss_history = [] # loss
# train_acc_history = [] # accuracy

# for epoch in range(30):

       
#     running_loss = 0.0
#     correct = 0.0
#     total = 0
    
#     # Iterating through the minibatches of the data
    
#     for i, data in enumerate(MyDataloader, 0):
        
#         # data is a tuple of (inputs, labels)
#         X, y = data
#         y = y-1
#         # print(y)
#         X = X.to(device)
#         y = y.long().to(device)
#         # y = y.to(device)
#         # Reset the parameter gradients  for the current minibatch iteration 
#         optimizer.zero_grad()

        
#         y_pred = model(X)             # Perform a forward pass on the network with inputs
#         loss = loss_fun(y_pred, y) # calculate the loss with the network predictions and ground Truth
#         loss.backward()             # Perform a backward pass to calculate the gradients
#         optimizer.step()            # Optimize the network parameters with calculated gradients

        
#         # Accumulate the loss and calculate the accuracy of predictions
#         running_loss += loss.item()
#         _, preds = torch.max(y_pred, 1) #convert output probabilities of each class to a singular class prediction
#         correct += preds.eq(y).sum().item()
#         total += y.size(0)
#         torch.cuda.empty_cache() # Don't know if its necessary

#         if (i + 1) % 5 == 0:
#             running_loss /= 5
#             correct /= total
#             print("[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%" % (epoch+1, i+1, running_loss, 100*correct))
#             train_loss_history.append(running_loss)
#             train_acc_history.append(correct)
#             running_loss = 0.0
#             correct = 0.0
#             total = 0
#     print("Epoch: ", epoch+1)



# plt.plot(train_acc_history)
# plt.plot(train_loss_history)
# plt.title("Dataset")
# plt.xlabel('iteration')
# plt.ylabel('acc/loss')
# plt.legend(['acc', 'loss'])
# plt.show()

# print('FINISH.')
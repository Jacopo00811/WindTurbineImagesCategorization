import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.transforms import v2 as transformsV2
from torch.utils.data import DataLoader
from Dataset import MyDataset

class MyNetwork(nn.Module):
    def __init__(self, input_channels):
        super(MyNetwork, self).__init__()
        self.input_channels = input_channels
        self.number_of_classes = 5

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
    

model = MyNetwork(input_channels=3)
BATCH_SIZE = 3
x = torch.randn(BATCH_SIZE, 3, 224, 224)
assert model(x).shape == torch.Size([BATCH_SIZE, 5])
print(model(x).shape)
total_params = sum(
	param.numel() for param in model.parameters()
)

print(f"Total number of parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fun = nn.CrossEntropyLoss()



root_directory = "WindTurbineImagesCategorization\\Data\\Dataset"
image_path = "WindTurbineImagesCategorization\\Data\\Dataset\\5\\20150709_D39M_IV.jpeg"
mode = "train"
split = {"train": 0.6, "val": 0.2, "test": 0.2}
mean = np.array([0.5750, 0.6065, 0.6459])
std = np.array([0.1854, 0.1748, 0.1794])

batch_size = 128
shuffle = True  
drop_last = False 
num_workers = 0 
transform = transformsV2.Compose([
    transformsV2.Resize((224, 224)), # Adjustable
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),  
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=mean.tolist(), std=std.tolist()),
    ]) 

# Create Dataset and Dataloader
Dataset = MyDataset(root_directory=root_directory, mode=mode, transform=transform, split=split)
print(f"Created a new Dataset of length: {len(Dataset)}")
MyDataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
print(f"Created a new Dataloader with batch size: {batch_size}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch = next(iter(MyDataloader))
# plot_batch_by_label(batch)
# plot_transformation_for_image_in_batch(image_path, mean, std)


# Initializing the list for storing the loss and accuracy

# train_loss_history = [] # loss
# train_acc_history = [] # accuracy

# for epoch in range(1):

       
#     running_loss = 0.0
#     correct = 0.0
#     total = 0
    
#     # Iterating through the minibatches of the data
    
#     for i, data in enumerate(MyDataloader, 0):
        
#         # data is a tuple of (inputs, labels)
#         X, y = data
#         y = y-1
#         print(y)
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

#         # Print statistics to console
#         if i % 1000 == 999: # print every 1000 mini-batches
#             running_loss /= 1000
#             correct /= total
#             print("[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%" % (epoch+1, i+1, running_loss, 100*correct))
#             train_loss_history.append(running_loss)
#             train_acc_history.append(correct)
#             running_loss = 0.0
#             correct = 0.0
#             total = 0

# print('FINISH.')
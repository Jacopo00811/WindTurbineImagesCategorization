import torch.nn as nn
import torch.nn.init as init

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


# def train_net(model, loss_function, device, dataloader_train, dataloader_val, optimizer, hyper_parameters, name="default"):
#     #def train_classifier(classifier, train_loader, val_loader, loss_func, tb_logger, epochs=10, name="default"):

#     # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs * len(train_loader) / 5, gamma=0.7) Not used in this exercise
#     epochs = hyper_parameters["epochs"]
#     validation_loss = 0
#     for epoch in range(epochs):  # loop over the dataset multiple times
        
#         """    Train step for one batch of data    """
#         training_loop = create_tqdm_bar(dataloader_train, desc=f'Training Epoch [{epoch+1}/{epochs}]')
#         training_loss = 0
#         #for i, data in enumerate(trainloader, 0):
#         for train_iteration, batch in training_loop:
#             model.train()  # Set the model to training mode
#             optimizer.zero_grad() # Reset the parameter gradients for the current minibatch iteration

#             images, labels = batch
#             labels -= 1 # Change the labels to start from 0

#             labels = labels.type(torch.LongTensor)
            
#             labels = labels.to(device)
#             images = images.to(device)

#             # forward + backward + optimize
#             predicted_labels = model(images)

#             loss_train = loss_function(predicted_labels, labels)
#             loss_train.backward()
#             optimizer.step()

#             # Accumulate the loss and calculate the accuracy of predictions
#             training_loss += loss_train.item()
#             # scheduler.step()

#             # _, preds = torch.max(y_pred, 1) #convert output probabilities of each class to a singular class prediction
#             # correct += preds.eq(y).sum().item()
#             # total += y.size(0)

#             # Update the progress bar.
#             training_loop.set_postfix(train_loss = "{:.8f}".format(training_loss / (train_iteration + 1)), val_loss = "{:.8f}".format(validation_loss))

#             # Update the tensorboard logger.
#             # tb_logger.add_scalar(f'{name}/train_loss', loss.item(), epoch * len(dataloader_train) + train_iteration)

#             # print statistics
#             # if train_iteration % 10 == 9:    # print every 10 batches
#             #     print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch+1,
#             #                                                        train_iteration+1,
#             #                                                        training_loss / (len(dataloader_train)*epoch+train_iteration)))
        

#         """    Validation step for one batch of data    """
#         val_loop = create_tqdm_bar(dataloader_val, desc=f'Validation Epoch [{epoch+1}/{epochs}]')
#         validation_loss = 0
#         with torch.no_grad():
#             for val_iteration, batch in val_loop:
#                 model.eval()  # Set the model to evaluation mode
#                 with torch.no_grad():
#                     images, labels = batch
#                     labels -= 1 # Change the labels to start from 0
#                     labels = labels.type(torch.LongTensor)
                    
#                     images = images.to(device)
#                     labels = labels.to(device)
#                     # Forward pass
#                     output = model(images)

#                     # Calculate the loss
#                     loss_val = loss_function(output, labels)

#                 validation_loss += loss_val.item()

#                 # Update the progress bar.
#                 val_loop.set_postfix(val_loss = "{:.8f}".format(validation_loss/(val_iteration+1)))

#                 # Update the tensorboard logger.
#                 # tb_logger.add_scalar(f'{name}/val_loss', validation_loss/(val_iteration+1), epoch*len(dataloader_val)+val_iteration)
            
#             # This value is for the progress bar of the training loop.
#             validation_loss /= len(dataloader_val)


# #i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
# path = os.path.join('logs', 'cls_logs')
# num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
# path = os.path.join(path, f'run_{num_of_runs + 1}')
# tb_logger = SummaryWriter(path)


# print("Finished training!")
#print("How did we do? Let's check the accuracy of the defaut classifier on the training and validation sets:")
# print(f"Training Acc: {classifier.getAcc(labled_train_loader)[1] * 100}%")
# print(f"Validation Acc: {classifier.getAcc(labled_val_loader)[1] * 100}%")
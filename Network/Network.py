import torch.nn as nn
import torch.nn.init as init

class MyNetwork(nn.Module):
    def __init__(self, hyper_parameters):
        super(MyNetwork, self).__init__()
        self.input_channels = hyper_parameters["input channels"]
        self.number_of_classes = hyper_parameters["number of classes"]
        self.pca = hyper_parameters.get("PCA", None)
        
        if self.pca:
            self.convolutional_layers = nn.Sequential(
                nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),            

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(), 
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),            

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(), 
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),            
            )
        else:
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

        if self.pca:
            self.fully_connected_layers = nn.Sequential(
                nn.Linear(12800, 2048),
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
        else:
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
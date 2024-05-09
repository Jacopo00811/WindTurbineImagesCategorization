from torch import nn
import torchvision
import os
from enum import Enum


# torch_home = 'C:\\Users\\jacop\\Desktop\\BSc\\Code\\WindTurbineImagesCategorization\\TransferLearning\\TorchvisionModels'
torch_home = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/TransferLearning/TorchvisionModels"
os.environ['TORCH_HOME'] = torch_home
os.makedirs(torch_home, exist_ok=True)
backbones = ["vgg19_bn", "mobilenet_v3_large", "resnet152"]

class FineTuneMode(Enum):
    """ Indicatea which layers we want to train during fine-tuning """
    " Just the new added layers " 
    NEW_LAYERS = 1
    " Just the classifier "
    CLASSIFIER = 2
    "Train all the layers "
    ALL_LAYERS = 3

class MultiModel(nn.Module):
    """ Custom class that wraps a torchvision model and provides methods to fine-tune """
    def __init__(self, backbone, hyperparameters, load_pretrained=False):
        super().__init__()
        assert backbone in backbones
        self.backbone = backbone
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        self.hyperparameters = hyperparameters

        if backbone == "mobilenet_v3_large":
            if load_pretrained:
                self.pretrained_model = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            else:
                self.pretrained_model = torchvision.models.mobilenet_v3_large(weights=None)
            
            self.classifier_layers = [self.pretrained_model.classifier]
            # Replace the final layer with a classifier for 5 classes
            self.pretrained_model.classifier[3] = nn.Linear(in_features=1280, out_features=self.hyperparameters["number of classes"], bias=True)
            self.new_layers = [self.pretrained_model.classifier[3]]
        elif backbone == "resnet152":
            if load_pretrained:
                self.pretrained_model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2)
            else:
                self.pretrained_model = torchvision.models.resnet152(weights=None)
            
            self.classifier_layers = [self.pretrained_model.fc]
            self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=self.hyperparameters["number of classes"], bias=True)
            self.new_layers = [self.pretrained_model.fc]
        elif backbone == "vgg19_bn":
            if load_pretrained:
                self.pretrained_model = torchvision.models.vgg19_bn(weights=torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1)
            else:
                self.pretrained_model = torchvision.models.vgg19_bn(weights=None)
            
            self.classifier_layers = [self.pretrained_model.classifier]
            self.pretrained_model.classifier[6] = nn.Linear(in_features=4096, out_features=self.hyperparameters["number of classes"], bias=True)
            self.new_layers = [self.pretrained_model.classifier[6]]

    def forward(self, x):
        return self.pretrained_model(x) # type: ignore
    
    def fine_tune(self, mode: FineTuneMode):
        " Fine-tune the model according to the specified mode using the requires_grad parameter "
        model = self.pretrained_model
        for parameter in model.parameters(): # type: ignore
            parameter.requires_grad = False

        if mode is FineTuneMode.NEW_LAYERS:
            for layer in self.new_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        elif mode is FineTuneMode.CLASSIFIER:
            for layer in self.classifier_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
        elif mode is FineTuneMode.ALL_LAYERS:
            for parameter in model.parameters(): # type: ignore
                parameter.requires_grad = True
        else:
            raise ValueError(f"Invalid mode: {mode}")

        print(f"Ready to fine-tune the model, with the {mode} set to train")

    def count_parameters(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        print(f"Total number of parameters: {total_params}")

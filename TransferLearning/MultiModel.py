import torch
from torch import nn
import torch.quantization
import numpy as np
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List, Any
from enum import Enum
from torchvision.datasets import VisionDataset
from typing import Tuple
from torch.utils.data import ConcatDataset

# TODO: Add directory and remove unsued imports

# torch_home = 'C:\\Users\\jacop\\Desktop\\BSc\\Code\\WindTurbineImagesCategorization\\TransferLearning\\TorchvisionModels'
torch_home = "/zhome/f9/0/168881/Desktop/WindTurbineImagesCategorization/TransferLearning/TorchvisionModels"
os.environ['TORCH_HOME'] = torch_home
os.makedirs(torch_home, exist_ok=True)
backbones = ["vgg19_bn", "mobilenet_v3_large", "resnet152"]

# classification_models = torchvision.models.list_models(module=torchvision.models)
# # print(len(classification_models), "classification models:", classification_models)


# vgg19_bn = torchvision.models.vgg19_bn(weights=None)
# mobilenet_v3_large = torchvision.models.mobilenet_v3_large(weights=None)
# resnet152 = torchvision.models.resnet152(weights=None)

# print("vgg19_bn\n", vgg19_bn.classifier)
# print("\nmobilenet_v3_large\n", mobilenet_v3_large.classifier)
# print("\nresnet152\n", resnet152.fc)
# print("\n\n")





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

    # def train_one_epoch(self, loader, optimizer, epoch):
    #     """Train this model for a single epoch. Return the loss computed
    #     during this epoch.
    #     """
    #     device = self.dummy_param.device
    #     criterion = nn.CrossEntropyLoss()
    #     running_loss = 0.0
    #     num_batches = 0

    #     for (inputs, targets) in iter(loader):
    #         optimizer.zero_grad()
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)

    #         outputs = self(inputs)
    #         loss = criterion(outputs, targets)

    #         running_loss, num_batches = running_loss + loss.item(), num_batches + 1
    #         loss.backward()
    #         optimizer.step()

    #     print(f"[{epoch}] Train Loss: {running_loss / num_batches:0.5f}")
    #     return running_loss / num_batches

    # def evaluate(self, loader, metric, epoch, run_type):
    #     """Evaluate the model on the specified dataset (provided using the DataLoader
    #     instance). Return the loss and accuracy.
    #     """
    #     device = self.dummy_param.device
    #     criterion = nn.CrossEntropyLoss()
    #     running_loss = 0.0
    #     running_accuracy = 0.0
    #     num_batches = 0
    #     for (inputs, targets) in iter(loader):
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)

    #         outputs = self(inputs)
    #         loss = criterion(outputs, targets)

    #         running_loss = running_loss + loss.item()
    #         num_batches = num_batches + 1
    #         running_accuracy += metric(outputs, targets).item()


    #     print(f"[{epoch}] {run_type} Loss: {running_loss / num_batches:.5f}, Accuracy: {running_accuracy / num_batches:.5f}")
    #     return running_loss / num_batches, running_accuracy / num_batches

    
    # def train_multiple_epochs_and_save_best_checkpoint(
    #     self,
    #     train_loader,
    #     val_loader,
    #     accuracy,
    #     optimizer,
    #     scheduler,
    #     epochs,
    #     filename,
    #     training_run,
    # ):
    #     """Train this model for multiple epochs. The caller is expected to have frozen
    #     the layers that should not be trained. We run training for "epochs" epochs.
    #     The model with the best val accuracy is saved after every epoch.
        
    #     After every epoch, we also save the train/val loss and accuracy.
    #     """
    #     best_val_accuracy = self.get_metrics("val")['accuracy']
    #     for epoch in range(1, epochs + 1):
    #         self.train()
    #         self.train_one_epoch(train_loader, optimizer, epoch)

    #         # Evaluate accuracy on the train dataset.
    #         self.eval()
    #         with torch.inference_mode():
    #             train_loss, train_acc = self.evaluate(train_loader, accuracy, epoch, "Train")
    #             training_run.train_loss.append(train_loss)
    #             training_run.train_accuracy.append(train_acc)
    #         # end with

    #         # Evaluate accuracy on the val dataset.
    #         self.eval()
    #         with torch.inference_mode():
    #             val_loss, val_acc = self.evaluate(val_loader, accuracy, epoch, "Val")
    #             training_run.val_loss.append(val_loss)
    #             training_run.val_accuracy.append(val_acc)
    #             if val_acc > best_val_accuracy:
    #                 # Save this checkpoint.
    #                 print(f"Current valdation accuracy {val_acc*100.0:.2f} is better than previous best of {best_val_accuracy*100.0:.2f}. Saving checkpoint.")
    #                 self.update_metrics("train", train_loss, train_acc)
    #                 self.update_metrics("val", val_loss, val_acc)
    #                 torch.save(self.state_dict(), filename)
    #                 best_val_accuracy = val_acc
    #         # end with
            
    #         scheduler.step()
    #     # end for (epoch)
    # # end def

#     def get_optimizer_params(self):
#         """This method is used only during model fine-tuning when we need to
#         set a linear or expotentially decaying learning rate (LR) for the
#         layers in the model. We exponentially decay the learning rate as we
#         move away from the last output layer.
#         """
#         options = []
#         if self.backbone == 'vgg19_bn':
#             # For vgg16, we start with a learning rate of 1e-3 for the last layer, and
#             # decay it to 1e-7 at the first conv layer. The intenmediate rates are
#             # decayed linearly.
#             lr = 0.0001
#             options.append({
#                 'params': self.pretrained_model.classifier.parameters(), # type: ignore
#                 'lr': lr,
#             })
#             final_lr = lr / 1000.0
#             diff_lr = final_lr - lr
#             lr_step = diff_lr / 44.0
#             for i in range(43, -1, -1):
#                 options.append({
#                     'params': self.pretrained_model.features[i].parameters(), # type: ignore
#                     'lr': lr + lr_step * (44-i)
#                 })
#             # end for
#         elif self.backbone in ['resnet50', 'resnet152']:
#             # For the resnet class of models, we decay the LR exponentially and reduce
#             # it to a third of the previos value at each step.
#             layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
#             lr = 0.0001
#             for layer_name in reversed(layers):
#                 options.append({
#                     "params": getattr(self.pretrained_model, layer_name).parameters(),
#                     'lr': lr,
#                 })
#                 lr = lr / 3.0
#             # end for
#         # end if
#         return options
#     # end def
# # end class

# # Sanity check to see if we can run a single forward pass with this model
# # when it is provided an input with the expected shape.
# for backbone in backbones:
#     print(f"Backbone: {backbone}")
#     fc_test = MultiModel(backbone=backbone, load_pretrained=False, hyperparameters={"number of classes": 5})
#     fc_test.count_parameters()
#     x = torch.randn(4, 3, 224, 224)
#     y = fc_test(x)
#     print(x.shape, y.shape)
#     # print(fc_test.get_optimizer_params())
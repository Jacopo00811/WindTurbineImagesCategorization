import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import v2 as transformsV2
import torch


class MyDataset(Dataset):
    """ 
    Custom dataset class for loading images from the given root directory, it 
    inherits from the torch.utils.data.Dataset class and implements the __len__ and
    __getitem__ methods.
    """

    def __init__(self, *args, root_directory, transform=None, mode="train", limit_files=None, 
                 split={"train": 0.6, "val": 0.2, "test": 0.2}, pca, **kwargs):
        super().__init__(*args, **kwargs)

        assert mode in ["train", "val", "test"], "wrong mode for dataset given"
        split_values = [v for _, v in split.items()]
        assert sum(split_values) == 1.0, "split values must sum up to 1"

        self.pca = pca # State variable to check if the dataset is PCA dataset
        self.root_directory = root_directory
        self.transform = transform
        self.mode = mode
        self.limit_files = limit_files
        self.split = split


        self.classes, self.class_to_idx = self._find_classes(self.root_directory)

        self.images, self.labels = self.make_dataset(
            directory=self.root_directory,
            class_to_idx=self.class_to_idx,
            mode=mode,
        )
 
        # Preload images
        # self.preloaded_images = [self.transform(self.load_image(path)) for path in self.images if self.load_image(path) is not None]  # type: ignore

        # mean, std = calculate_mean_and_std(self.root_directory) hardcoded for speed
        self.test_transform = transformsV2.Compose([ 
            transformsV2.Resize((224, 224)),
            transformsV2.ToImage(), 
            transformsV2.ToDtype(torch.float32, scale=True),
            # transformsV2.Normalize(mean=[0.5750, 0.6065, 0.6459], std=[0.1854, 0.1748, 0.1794]),
        ])


        self.val_transform = transformsV2.Compose([
            transformsV2.Pad(300, padding_mode="reflect"),
            transformsV2.RandomHorizontalFlip(p=0.5),
            transformsV2.RandomVerticalFlip(p=0.5),
            transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transformsV2.RandomAutocontrast(p=0.5),
            transformsV2.RandomRotation(degrees=[0, 90]),
            transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
            # Replace deprecated ToTensor()
            transformsV2.CenterCrop(224),
            transformsV2.Resize((224, 224)), # Adjustable
            transformsV2.ToImage(),
            transformsV2.ToDtype(torch.float32, scale=True),
            # transformsV2.Normalize(mean=[0.5750, 0.6065, 0.6459], std=[0.1854, 0.1748, 0.1794]),
        ])


    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [folder.name for folder in os.scandir(directory) if folder.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i+1 for i in range(len(classes))} # Changed the index to start from 1 as actual labels
        return classes, class_to_idx

    def select_split(self, images, labels, mode):
        """
        Depending on the mode of the dataset, deterministically split it.

        :param images, a list containing paths to all images in the dataset
        :param labels, a list containing one label per image

        :returns (images, labels), where only the indices for the corresponding data split are selected.
        """
        fraction_train = self.split["train"]
        fraction_val = self.split["val"]
        num_samples = len(images)
        num_train = int(num_samples * fraction_train)
        num_valid = int(num_samples * fraction_val)

        np.random.seed(0)
        rand_perm = np.random.permutation(num_samples)

        if mode == "train":
            idx = rand_perm[:num_train] # type: ignore
        elif mode == "val":
            idx = rand_perm[num_train:num_train+num_valid]
        elif mode == "test":
            idx = rand_perm[num_train+num_valid:]

        if self.limit_files:
            idx = idx[:self.limit_files]

        if isinstance(images, list):
            return list(np.array(images)[idx]), list(np.array(labels)[idx])
        else:
            return images[idx], list(np.array(labels)[idx])

    def make_dataset(self, directory, class_to_idx, mode):
        """
        Create the image dataset by preparaing a list of samples
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (images, labels) where:
            - images is a list containing paths to all images in the dataset
            - labels is a list containing one label per image
        """
        images, labels = [], []
        for target_class in sorted(class_to_idx.keys()):
            label = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(label)

        images, labels = self.select_split(images, labels, mode)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        length = len(self.images)
        return length

    @staticmethod
    def load_image(image_path):
        """Load image from image_path as Image object using PIL.Image.open()"""
        if image_path.endswith(".png"):
            return Image.open(image_path)
        elif image_path.endswith(".pt"):
            return torch.load(image_path)
        else:
            raise ValueError("Unknown file type for image")
    
    def __getitem__(self, index):
        label = self.labels[index]
        path = self.images[index]
        image = self.load_image(path)
        
        if self.transform is not None and self.mode == "train" and not self.pca:
            image = self.transform(image)
        elif self.transform is None and self.mode == "train" and self.pca: # PCA dataset 
            pass # Do nothing
        elif self.transform is None and self.mode == "val" and not self.pca:
            image = self.val_transform(image)
        elif self.transform is None and self.mode == "test" and not self.pca:
            image = self.test_transform(image)
        elif self.transform is not None and self.mode == "val" and self.pca:
            pass # Do nothing
        elif self.transform is not None and self.mode == "test" and self.pca:
            pass # Do nothing

        return image, label

    # def get_preloaded_images(self):
    #     return self.preloaded_images
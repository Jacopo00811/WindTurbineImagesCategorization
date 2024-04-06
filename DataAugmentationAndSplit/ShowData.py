from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2 as transformsV2
from Dataset import MyDataset
import torch
from PIL import Image

def main():
    root_directory = "Data\\Dataset"
    image_path = "Data\\Dataset\\5\\20150709_D39M_IV.jpeg"
    mode = "train"
    split = {"train": 0.6, "val": 0.2, "test": 0.2}
    mean = np.array([0.5750, 0.6065, 0.6459])
    std = np.array([0.1854, 0.1748, 0.1794])

    batch_size = 16
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

    batch = next(iter(MyDataloader))
    plot_batch_by_label(batch)
    plot_transformation_for_image_in_batch(image_path, mean, std)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for images, _ in loader:
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

def plot_batch_by_label(batch):
    """
    Plots a batch of images grouped by their labels in separate columns.

    Args:
    - batch (dict): A dictionary containing 'image' and 'label' keys.
    - mean (list): Mean values used for normalization.
    - std (list): Standard deviation values used for normalization.
    """
    images, labels = batch
    
    # inv_normalize = transformsV2.Normalize(mean=-mean/std, std=1/std) # Inverse normalization transform to view images with original colors
    # images = inv_normalize(images)

    # Convert the batch of tensor (NxCXHxW) to a tensor (NXHxWxC). Also clip and recale values to [0, 1]
    images_ready = images.permute(0, 2, 3, 1)
    labels_np = labels.numpy()
    
    unique_labels, counts = np.unique(labels_np, return_counts=True)

    num_labels = len(labels_np)
    num_images = images_ready.shape[0]

    assert num_labels == num_images, "Number of labels and images must be equal."

    # Calculate the number of rows and columns for subplots
    num_rows = max(counts)
    num_cols = len(unique_labels)

    _, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10), sharex=True, sharey=True)

    for i, label in enumerate(unique_labels):
        label_images = images_ready[labels_np == label]
        num_images_label = label_images.shape[0]
        for j, image in enumerate(label_images):
            axs[j, i].imshow(rescale_0_1(image))
            axs[j, i].axis("off")
        for j in range(num_images_label, num_rows):  # Hide empty subplots
            axs[j, i].axis("off")
        axs[0, i].set_title(f"Label: {label}", fontweight="bold")
    
    plt.suptitle(f"Images grouped by labels for batch of size {len(images)}", fontsize=20, fontweight="bold", color="red")
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))  # Adjust layout to accommodate title
    
    plt.tight_layout()
    plt.show()

def plot_transformation_for_image_in_batch(image_path, mean, std):
    """
    Plots all the transformation that have been applied to batch on a sample image.

    Args:
    - image path (str): Path to the image file.
    - mean (list): Mean values used for normalization.
    - std (list): Standard deviation values used for normalization.
    """
    transforms = [
        transformsV2.RandomHorizontalFlip(p=1),
        transformsV2.RandomVerticalFlip(p=1),
        transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=1),
        transformsV2.RandomAutocontrast(p=1),  
        transformsV2.RandomRotation(degrees=[0, 90]),
        transformsV2.ColorJitter(brightness=0.25, contrast=0.20), 
        transformsV2.Normalize(mean=mean, std=std)] 
    
    transform_names = [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomAdjustSharpness",
        "RandomAutocontrast",
        "RandomRotation",
        "ColorJitter",
        "Normalize"
    ]   

    toTensor = transformsV2.Compose([
        transformsV2.Resize((249, 249)),
        transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
        transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    ])

    image = Image.open(image_path)
    image = toTensor(image)
    image_ready = image.permute(1, 2, 0)    
    _, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    axes[0].imshow(image_ready)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    for i, transform in enumerate(transforms):
        ax = axes[i+1]  # Calculate index for subplot
        transformed_image = transform(image_ready.permute(2, 0, 1)).permute(1, 2, 0)
        if i == 6:
            transformed_image = rescale_0_1(transformed_image)
        
        ax.imshow(transformed_image)
        ax.set_title(f"{transform_names[i]}", fontweight="bold")
        ax.axis("off")
    
    plt.suptitle("Possible transformations on a sample image", fontsize=20, fontweight="bold", color="red")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))  # Adjust layout to accommodate title
    
    plt.tight_layout()
    plt.show()

def rescale_0_1(image):
    """Rescale pixel values to range [0, 1] for visualization purposes only."""
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image-min_val)/abs(max_val-min_val)
    return rescaled_image


if __name__ == "__main__":
    main()

# # Create Dataset and Dataloader
# Dataset = MyDataset(root_directory=root_directory, mode=mode, transform=transform, split=split)
# print(f"Created a new Dataset of length: {len(Dataset)}")
# MyDataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
# print(f"Created a new Dataloader with batch size: {batch_size}")

# batch = next(iter(MyDataloader))
# plot_batch_by_label(batch)
# plot_transformation_for_image_in_batch(image_path, mean, std)


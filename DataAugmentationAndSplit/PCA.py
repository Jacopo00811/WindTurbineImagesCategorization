import torch
from torchvision.transforms import v2 as transformsV2
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Network.Dataset import MyDataset
from ShowData import rescale_0_1

# Perform PCA on the dataset and save the reduced images
def PCA_on_dataset_v2(root_directory, save_root, transform, split, COMPONENTS):
    mode = ["train", "val", "test"]
    pca = PCA(n_components=COMPONENTS)
    for mode in mode:
        if mode == "train":
            dataset_train = MyDataset(root_directory=root_directory, mode=mode, transform=transform, split=split, pca=False)
            dataLoader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            all = np.vstack([img.reshape(img.shape[0], -1).numpy() for img, _ in dataLoader_train])
            labels = [label.item() for _, label in dataLoader_train]
            pca.fit(all)
            reduced_images = torch.from_numpy(pca.transform(all).reshape(1320, 3, 5, 5))
            for i in range(reduced_images.shape[0]):
                label_folder = os.path.join(save_root, str(labels[i]))
                os.makedirs(label_folder, exist_ok=True)
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(reduced_images[i], file_path)
                print("Saved image with max value: ", reduced_images[i].max(), "and min value: ", reduced_images[i].min())
            print("\n PCA on train dataset completed\n")

        elif mode == "val":
            dataset_val = MyDataset(root_directory=root_directory, mode=mode, transform=None, split=split, pca=False)
            dataLoader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            all = np.vstack([img.reshape(img.shape[0], -1).numpy() for img, _ in dataLoader_val])
            labels = [label.item() for _, label in dataLoader_val]
            reduced_images = torch.from_numpy(pca.transform(all).reshape(440, 3, 5, 5))
            for i in range(reduced_images.shape[0]):
                label_folder = os.path.join(save_root, str(labels[i]))
                os.makedirs(label_folder, exist_ok=True)
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(reduced_images[i], file_path)
                print("Saved image with max value: ", reduced_images[i].max(), "and min value: ", reduced_images[i].min())
            print("\n PCA on validation dataset completed\n")

        elif mode == "test":
            dataset_test = MyDataset(root_directory=root_directory, mode=mode, transform=None, split=split, pca=False)
            dataLoader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            all = np.vstack([img.reshape(img.shape[0], -1).numpy() for img, _ in dataLoader_test])
            labels = [label.item() for _, label in dataLoader_test]
            reduced_images = torch.from_numpy(pca.transform(all).reshape(440, 3, 5, 5))
            for i in range(reduced_images.shape[0]):
                label_folder = os.path.join(save_root, str(labels[i]))
                os.makedirs(label_folder, exist_ok=True)
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(reduced_images[i], file_path)
                print("Saved image with max value: ", reduced_images[i].max(), "and min value: ", reduced_images[i].min())
            print("\n PCA on test dataset completed\n")
        else:
            raise ValueError("Invalid mode. Choose between 'train', 'val' and 'test'")
    print("\n\n################## PCA on dataset completed ##################\n\n")

# Create plots for the original image, image channels, explained variance ratios, first three principal components, original image and reduced image
def show_PCA_for_sample_v2(image_path, components, transform, root_directory, split):
    image = Image.open(image_path)
    image = transform(image)
    image = rescale_0_1(image)
    print(f"Image size: {image.shape}")
    dataset_train = MyDataset(root_directory=root_directory, mode="train", transform=transform, split=split, pca=False)
    dataLoader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
    all_images = np.vstack([img.reshape(img.shape[0], -1).numpy() for img, _ in dataLoader_train])
    print(f"Dataset size: {all_images.shape}")
    pca = PCA(n_components=components)
    pca.fit(all_images)
    
    red_channel = image[2, :, :]
    green_channel = image[1, :, :]
    blue_channel = image[0, :, :]

    # Plotting the original image
    plt.title("Original Image", fontweight='bold')
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    
    # Plotting the image channels
    fig = plt.figure(figsize = (12, 7)) 
    fig.add_subplot(131)
    plt.title("Blue Channel", fontweight='bold')
    plt.imshow(blue_channel)
    plt.axis('off')
    fig.add_subplot(132)
    plt.title("Green Channel", fontweight='bold')
    plt.imshow(green_channel)
    plt.axis('off')
    fig.add_subplot(133)
    plt.title("Red Channel", fontweight='bold')
    plt.imshow(red_channel)
    plt.axis('off')
    fig.suptitle("Image Channels", fontweight='bold', fontsize=20, color='red')
    plt.show()

    # Plotting the explained variance ratios V1
    plt.title("PCA Analysis", fontweight='bold')
    plt.xlabel("Number of components", fontweight='bold')
    plt.ylabel("Cumulative explained variance", fontweight='bold')
    plt.plot(np.cumsum(pca.explained_variance_ratio_), linestyle='-', color='blue', marker='x', markeredgecolor='black')
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.xticks(np.arange(0, components+1, 5))
    plt.show()

    # Plotting the explained variance ratios V2
    plt.title("PCA Analysis", fontweight='bold')
    plt.ylabel('Variation explained', fontweight='bold')
    plt.xlabel('Eigenvalues', fontweight='bold')
    plt.bar(list(range(1, components+1)), pca.explained_variance_ratio_, color='blue', label=f'Explained Variance Ratio ({np.sum(pca.explained_variance_ratio_):.4f})')
    plt.legend()
    plt.show()

    # Plotting the first three principal components
    fig = plt.figure(figsize = (12, 7))
    for i in range(3):
        fig.add_subplot(131+i)
        n_th_eigenvector = rescale_0_1(torch.from_numpy(pca.components_[i, :].reshape(image.shape[0], image.shape[1], image.shape[2])))
        plt.imshow(n_th_eigenvector.permute(1, 2, 0))
        plt.title(f"Principal Component {i+1}", fontweight='bold')
        plt.axis('off')
    fig.suptitle("First Three Principal Components", fontweight='bold', fontsize=20, color='red')
    plt.show()
    
    image = image.unsqueeze(0)
    reduced_image = pca.transform(image.reshape(image.shape[0], -1).numpy())
    # transformed_image = torch.from_numpy(pca.inverse_transform(reduced_image)).reshape(1, 3, 224, 224)
    print(f"Reduce image size: {reduced_image.shape}")
    reduced_image = torch.from_numpy(reduced_image.reshape(1, 3, 5, 5))
    print(f"Reduce image size after reshape: {reduced_image.shape}")

    fig = plt.figure(figsize = (10, 5)) 
    fig.add_subplot(121)
    plt.title("Original Image", fontweight='bold')
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    fig.add_subplot(122)
    plt.title("Reduced Image", fontweight='bold')
    plt.imshow(reduced_image.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    # fig.add_subplot(133)
    # plt.title("Transformed Image", fontweight='bold')
    # plt.imshow(transformed_image.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    fig.suptitle("Image Comparison", fontweight='bold', fontsize=20, color='red')
    plt.show()








COMPONENTS = 75
split = {"train": 0.6, "val": 0.2, "test": 0.2}
MEAN = np.array([0.5750, 0.6065, 0.6459])
STD = np.array([0.1854, 0.1748, 0.1794])
root_directory = "WindTurbineImagesCategorization\\Data\\DatasetPNG_1"
save_root = "WindTurbineImagesCategorization\\Data\\Test"

transform = transformsV2.Compose([
    transformsV2.Pad(300, padding_mode="reflect"),
    transformsV2.RandomHorizontalFlip(p=0.5),
    transformsV2.RandomVerticalFlip(p=0.5),
    transformsV2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transformsV2.RandomAutocontrast(p=0.5),  
    transformsV2.RandomRotation(degrees=[0, 90]),
    transformsV2.ColorJitter(brightness=0.25, saturation=0.20),
    transformsV2.CenterCrop(224),
    transformsV2.Resize((224, 224)), # Adjustable
    transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
    transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 
path = "WindTurbineImagesCategorization\\Data\\DatasetPNG\\5\\20180328_C9WC_IV.png"
show_PCA_for_sample_v2(path, COMPONENTS, transform, root_directory, split)
# PCA_on_dataset_v2(root_directory, save_root, transform, split, COMPONENTS)








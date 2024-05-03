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


def PCA_for_image(image, components):
    transformed_channels = torch.zeros(image.shape[0], image.shape[1], components)
    explained_var_ratios = np.array([])

    for channel_idx in range(image.shape[0]):
        channel = image[channel_idx, :, :]
        pca = PCA(n_components=components)
        pca.fit(channel)
        trans_pca = pca.transform(channel)
        explained_var_ratio = sum(pca.explained_variance_ratio_)
        # transformed_channel = torch.from_numpy(pca.inverse_transform(trans_pca))
        # transformed_channels[channel_idx, :, :] = transformed_channel
        transformed_channels[channel_idx, :, :] = torch.from_numpy(trans_pca)
        explained_var_ratios = np.append(explained_var_ratios, explained_var_ratio)

    return transformed_channels, explained_var_ratios

def PCA_on_dataset(root_directory, save_root, transform, split, COMPONENTS):
    mode = ["train", "val", "test"]
    pca = PCA(n_components=COMPONENTS)
    for mode in mode:
        if mode == "train":
            dataset_train = MyDataset(root_directory=root_directory, mode=mode, transform=transform, split=split, pca=False)
            dataLoader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            for i, (image, label) in enumerate(dataLoader_train):
                label_folder = os.path.join(save_root, str(label.item()))
                os.makedirs(label_folder, exist_ok=True)
                image = image.squeeze(0) # Remove batch dimension
                transformed_channels_train = torch.zeros(image.shape[0], image.shape[1], COMPONENTS)
                for channel_idx in range(image.shape[0]):
                    channel = image[channel_idx, :, :]
                    pca.fit(channel)
                    trans_pca = pca.transform(channel)
                    # transformed_channel = torch.from_numpy(pca.inverse_transform(trans_pca))
                    transformed_channels_train[channel_idx, :, :] = torch.from_numpy(trans_pca) # transformed_channel
                # Save transformed image
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(transformed_channels_train, file_path)
                print("Saved image with max value: ", transformed_channels_train.max(), "and min value: ", transformed_channels_train.min())
            print("\n PCA on train dataset completed\n")
        elif mode == "val":
            dataset_val = MyDataset(root_directory=root_directory, mode=mode, transform=None, split=split, pca=False)
            dataLoader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            for i, (image, label) in enumerate(dataLoader_val):
                label_folder = os.path.join(save_root, str(label.item()))
                os.makedirs(label_folder, exist_ok=True)
                image = image.squeeze(0)
                transformed_channels_val = torch.zeros(image.shape[0], image.shape[1], COMPONENTS)
                for channel_idx in range(image.shape[0]):
                    channel = image[channel_idx, :, :]
                    trans_pca = pca.transform(channel)
                    # transformed_channel = torch.from_numpy(pca.inverse_transform(trans_pca))
                    transformed_channels_val[channel_idx, :, :] = torch.from_numpy(trans_pca) # transformed_channel
                # Save transformed image
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(transformed_channels_train, file_path)
                print("Saved image with max value: ", transformed_channels_train.max(), "and min value: ", transformed_channels_train.min())
            print("\n PCA on validation dataset completed\n")
        elif mode == "test":
            dataset_test = MyDataset(root_directory=root_directory, mode=mode, transform=None, split=split, pca=False)
            dataLoader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            for i, (image, label) in enumerate(dataLoader_test):
                label_folder = os.path.join(save_root, str(label.item()))
                os.makedirs(label_folder, exist_ok=True)
                image = image.squeeze(0)
                transformed_channels_test = torch.zeros(image.shape[0], image.shape[1], COMPONENTS)
                for channel_idx in range(image.shape[0]):
                    channel = image[channel_idx, :, :]
                    trans_pca = pca.transform(channel)
                    # transformed_channel = torch.from_numpy(pca.inverse_transform(trans_pca))
                    transformed_channels_test[channel_idx, :, :] = torch.from_numpy(trans_pca) # transformed_channel
                # Save transformed image
                filename = f"image_{mode}_{i}.pt"
                file_path = os.path.join(label_folder, filename)
                torch.save(transformed_channels_train, file_path)
                print("Saved image with max value: ", transformed_channels_train.max(), "and min value: ", transformed_channels_train.min())
            print("\n PCA on test dataset completed\n")
        else:
            raise ValueError("Invalid mode. Choose between 'train', 'val' and 'test'")
    print("\n\n################## PCA on dataset completed ##################\n\n")

def process_dataset(dataLoader, save_root, COMPONENTS):
    dataset_explained_var_ratios = np.array([])
    for i, (image, label) in enumerate(dataLoader):
        label_folder = os.path.join(save_root, str(label.item()))
        os.makedirs(label_folder, exist_ok=True)

        image = image.squeeze(0) # Remove batch dimension

        image, explained_var_ratios = PCA_for_image(image, COMPONENTS)
        dataset_explained_var_ratios = np.append(dataset_explained_var_ratios, explained_var_ratios)
        
        filename = f"image_{i}.png"
        file_path = os.path.join(label_folder, filename)
        save_image(image, file_path)
        print(f"Saved image {i} to {file_path} with max value {image.max()} and min value {image.min()}")
        break
    return dataset_explained_var_ratios

def show_PCA_for_sample(image_path, components, transform):
    image = Image.open(image_path)
    image = transform(image).permute(1, 2, 0)
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    # Plotting the original image
    plt.title("Original Image", fontweight='bold', fontsize=20, color='red')
    plt.imshow(image)
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

    pca_b = PCA(n_components=components)
    pca_b.fit(blue_channel)
    trans_pca_b = pca_b.transform(blue_channel)
    pca_g = PCA(n_components=components)
    pca_g.fit(green_channel)
    trans_pca_g = pca_g.transform(green_channel)
    pca_r = PCA(n_components=components)
    pca_r.fit(red_channel)
    trans_pca_r = pca_r.transform(red_channel)

    # Plotting the explained variance ratios V1
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(131)
    ax1.set_title("Blue Channel", fontweight='bold')
    ax1.set_ylabel('Cumulative explained variance')
    ax1.set_xlabel('Number of components')
    ax1.plot(np.cumsum(pca_b.explained_variance_ratio_), linestyle='-', color='blue', marker='x', markeredgecolor='black')
    ax1.grid(True)
    ax1.set_yticks(np.arange(0, 1.1, 0.05))
    ax1.set_xticks(np.arange(0, components+1, 5))
    ax2 = fig.add_subplot(132)
    ax2.set_title("Green Channel", fontweight='bold')
    ax2.set_ylabel('Cumulative explained variance')
    ax2.set_xlabel('Number of components')
    ax2.plot(np.cumsum(pca_g.explained_variance_ratio_), linestyle='-', color='green', marker='x', markeredgecolor='black')
    ax2.grid(True)
    ax2.set_yticks(np.arange(0, 1.1, 0.05))
    ax2.set_xticks(np.arange(0, components+1, 5))
    ax3 = fig.add_subplot(133)
    ax3.set_title("Red Channel", fontweight='bold')
    ax3.set_ylabel('Cumulative explained variance')
    ax3.set_xlabel('Number of components')
    ax3.plot(np.cumsum(pca_r.explained_variance_ratio_), linestyle='-', color='red', marker='x', markeredgecolor='black')
    ax3.grid(True)
    ax3.set_yticks(np.arange(0, 1.1, 0.05))
    ax3.set_xticks(np.arange(0, components+1, 5))
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.suptitle("PCA Analysis of Image Channels", fontweight='bold', fontsize=20, color='red')
    plt.show()

    # Plotting the explained variance ratios V2
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(131)
    ax1.set_title("Blue Channel", fontweight='bold')
    ax1.set_ylabel('Variation explained')
    ax1.set_xlabel('Eigen Value')
    ax1.bar(list(range(1, components+1)), pca_b.explained_variance_ratio_, color='blue', label=f'Explained Variance Ratio ({np.sum(pca_b.explained_variance_ratio_):.4f})')
    ax1.legend()
    ax2 = fig.add_subplot(132)
    ax2.set_title("Green Channel", fontweight='bold')
    ax2.set_ylabel('Variation explained')
    ax2.set_xlabel('Eigen Value')
    ax2.bar(list(range(1, components+1)), pca_g.explained_variance_ratio_, color='green', label=f'Explained Variance Ratio ({np.sum(pca_g.explained_variance_ratio_):.4f})')
    ax2.legend()
    ax3 = fig.add_subplot(133)
    ax3.set_title("Red Channel", fontweight='bold')
    ax3.set_ylabel('Variation explained')
    ax3.set_xlabel('Eigen Value')
    ax3.bar(list(range(1, components+1)), pca_r.explained_variance_ratio_, color='red', label=f'Explained Variance Ratio ({np.sum(pca_r.explained_variance_ratio_):.4f})')
    ax3.legend()
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.suptitle("PCA Analysis of Image Channels", fontweight='bold', fontsize=20, color='red')
    plt.show()

    b_tensor = torch.from_numpy(pca_b.inverse_transform(trans_pca_b))
    g_tensor = torch.from_numpy(pca_g.inverse_transform(trans_pca_g))
    r_tensor = torch.from_numpy(pca_r.inverse_transform(trans_pca_r))
    merged_tensor = torch.stack((b_tensor, g_tensor, r_tensor), dim=2)

    fig = plt.figure(figsize = (9, 7)) 
    fig.add_subplot(121)
    plt.title("Original Image", fontweight='bold')
    plt.imshow(image)
    plt.axis('off')
    fig.add_subplot(122)
    plt.title("Reduced Image", fontweight='bold')
    plt.imshow(merged_tensor)
    plt.axis('off')
    fig.suptitle("Image Comparison", fontweight='bold', fontsize=20, color='red')
    plt.show()

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

def show_PCA_for_sample_v2(image_path, components, transform, root_directory, split):
    image = Image.open(image_path)
    image = transform(image)

    dataset_train = MyDataset(root_directory=root_directory, mode="train", transform=transform, split=split, pca=False)
    dataLoader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    all_images = np.vstack([img.reshape(img.shape[0], -1).numpy() for img, _ in dataLoader_train])
    
    pca = PCA(n_components=components)
    pca.fit(all_images)
    
    red_channel = image[2, :, :]
    green_channel = image[1, :, :]
    blue_channel = image[0, :, :]

    # Plotting the original image
    plt.title("Original Image", fontweight='bold', fontsize=20, color='red')
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
    plt.title("PCA Analysis", fontweight='bold', fontsize=20, color='red')
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.plot(np.cumsum(pca.explained_variance_ratio_), linestyle='-', color='blue', marker='x', markeredgecolor='black')
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.xticks(np.arange(0, components+1, 5))
    plt.show()

    # Plotting the explained variance ratios V2
    plt.title("PCA Analysis", fontweight='bold', fontsize=20, color='red')
    plt.ylabel('Variation explained')
    plt.xlabel('Eigen Value')
    plt.bar(list(range(1, components+1)), pca.explained_variance_ratio_, color='blue', label=f'Explained Variance Ratio ({np.sum(pca.explained_variance_ratio_):.4f})')
    plt.legend()
    plt.show()

    image = image.unsqueeze(0)
    reduced_image = pca.transform(image.reshape(image.shape[0], -1).numpy())
    # transformed_image = torch.from_numpy(pca.inverse_transform(reduced_image)).reshape(1, 3, 224, 224)
    print(f"Reduce image size: {reduced_image.shape}")
    reduced_image = torch.from_numpy(reduced_image.reshape(1, 3, 5, 5))
    print(f"Reduce image sizeafter reshape: {reduced_image.shape}")

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
    # transformsV2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
    ]) 

# show_PCA_for_sample_v2("WindTurbineImagesCategorization\\Data\\DatasetPNG\\4\\20140611_C4HY_II.png", COMPONENTS, transform, root_directory, split)
PCA_on_dataset_v2(root_directory, save_root, transform, split, COMPONENTS)








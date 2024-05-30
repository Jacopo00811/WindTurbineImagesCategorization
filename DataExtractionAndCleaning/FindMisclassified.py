import os
import shutil
import PIL.Image
from torchvision.transforms import v2 as transformsV2
import torch
import re


def extract_tensor_numbers(input_string):
    # Regular expression to find the numbers inside the brackets
    pattern = r'tensor\(\[([\d, ]+)\]'

    match = re.search(pattern, input_string)
    if match:
        # Extract the matched group (the numbers inside the brackets)
        numbers_str = match.group(1)
        numbers_list = numbers_str.split(', ')
        return numbers_list
    else:
        return []
    
def filterWrongClassified(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    new_directory = os.path.join(directory, "Misclassified")
    os.makedirs(new_directory, exist_ok=True)

    for file_name in files:
        true_label = file_name.split('_')[4]
        predicted_label = file_name.split('_')[6].split('.')[0]
        if true_label != predicted_label:
            print(f'The picture {file_name} has been misclassified')
            shutil.move(os.path.join(directory, file_name), os.path.join(new_directory, file_name)) # type: ignore

def filterVeryOff(directory_misclassified):
    files = [f for f in os.listdir(directory_misclassified) if os.path.isfile(os.path.join(directory_misclassified, f))]

    new_directory = os.path.join(directory_misclassified, "VeryMisclassified")
    os.makedirs(new_directory, exist_ok=True)

    for file_name in files:
        true_label = file_name.split('_')[4]
        predicted_label = file_name.split('_')[6].split('.')[0]
        if abs(int(true_label) - int(predicted_label)) > 1:
            print(f'The picture {file_name} is very misclassified with a difference of {abs(int(true_label) - int(predicted_label))}')
            shutil.move(os.path.join(directory_misclassified, file_name), os.path.join(new_directory, file_name)) # type: ignore

def rescale_0_1(image):
    """Rescale pixel values to range [0, 1] for visualization purposes only."""
    min_val = image.min()
    max_val = image.max()
    rescaled_image = (image-min_val)/abs(max_val-min_val)
    return rescaled_image

def rescale_dataset(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    toTensor = transformsV2.Compose([
        transformsV2.Resize((224, 224)),
        transformsV2.ToImage(),                          # Replace deprecated ToTensor()    
        transformsV2.ToDtype(torch.float32, scale=True), # Replace deprecated ToTensor() 
    ])
    for file_name in files:
        image = PIL.Image.open(os.path.join(directory, file_name))
        
        image = toTensor(image)
        print(f"Min: {image.min()}, Max: {image.max()}")
        image = rescale_0_1(image)
        print(f"New Min: {image.min()}, New Max: {image.max()}")
        image = transformsV2.ToPILImage()(image)
        os.remove(os.path.join(directory, file_name))
        image.save(os.path.join(directory, file_name))


def filterWrongClassified_V2(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    new_directory = os.path.join(directory, "Misclassified")
    os.makedirs(new_directory, exist_ok=True)

    for file_name in files:
        true_label = file_name.split('_')[4]
        predicted_labels = extract_tensor_numbers(file_name.split('_')[6])
        print(predicted_labels)
        if true_label not in predicted_labels:
            print(f'The picture {file_name} has been misclassified')
            shutil.move(os.path.join(directory, file_name), os.path.join(new_directory, file_name)) # type: ignore



filterWrongClassified_V2("WindTurbineImagesCategorization\\Data\\Test")

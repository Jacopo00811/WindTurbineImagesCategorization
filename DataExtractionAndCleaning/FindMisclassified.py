import os
import shutil

def filterWrongClassified(directory):

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    new_directory = os.makedirs(os.path.join(directory, "Misclassified"), exist_ok=True)

    for file_name in files:
        true_label = file_name.split('_')[4]
        predicted_label = file_name.split('_')[6].split('.')[0]
        if true_label != predicted_label:
            print(f'The picture {file_name} has been misclassified')
            shutil.move(os.path.join(directory, file_name), os.path.join(new_directory, file_name)) # type: ignore


def filterVeryOff(directory_misclassified):
    files = [f for f in os.listdir(directory_misclassified) if os.path.isfile(os.path.join(directory_misclassified, f))]

    new_directory = os.makedirs(os.path.join(directory_misclassified, "VeryMisclassified"), exist_ok=True)

    for file_name in files:
        true_label = file_name.split('_')[4]
        predicted_label = file_name.split('_')[6].split('.')[0]
        if abs(int(true_label) - int(predicted_label)) > 1:
            print(f'The picture {file_name} is very misclassified with a difference of {abs(int(true_label) - int(predicted_label))}')
            shutil.move(os.path.join(directory_misclassified, file_name), os.path.join(new_directory, file_name)) # type: ignore

import os
import shutil
import random

source_folder = "Data\\Dataset\\1"
desired_count = 440

image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

existing_count = len(image_files)
if existing_count >= desired_count:
    print("Error: There are already enough images in the source folder.")
    exit()

random.shuffle(image_files)
copied_count = existing_count

while copied_count < desired_count:
    random_image = random.choice(image_files)
    source_path = os.path.join(source_folder, random_image)
    destination_path = os.path.join(source_folder, f"duplicate_{copied_count+1}_{random_image}")
    shutil.copyfile(source_path, destination_path)    
    copied_count += 1

print(f"{copied_count} images copied and duplicated in {source_folder}.")

import re
import os

def remove_img_prefix(filename):
    # Remove the "imgNUMBER_" prefix from the filename and add an underscore between the number and the rest of ID 
    return re.sub(r'img(\d+)_(\d+) (.+)', r'\2_\3', filename)

folder_path = "Data\\Test"

for filename in os.listdir(folder_path):
    new_filename = remove_img_prefix(filename)
    if new_filename in os.listdir(folder_path):
        print(f"Duplicate found: {filename} -> {new_filename}")
        
    # Join the folder path with the old and new filenames
    old_filepath = os.path.join(folder_path, filename)
    new_filepath = os.path.join(folder_path, new_filename)
    # Rename the file
    os.rename(old_filepath, new_filepath)
    print(f"{filename} -> {new_filename}")
from PIL import Image
import os

def crop_images(input_folder, output_folder, pixels_to_crop):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        image_path = os.path.join(input_folder, file)
        img = Image.open(image_path)
        width, height = img.size
        left = pixels_to_crop
        top = pixels_to_crop
        right = width - pixels_to_crop
        bottom = height - pixels_to_crop
        
        cropped_img = img.crop((left, top, right, bottom))

        output_path = os.path.join(output_folder, file)
        cropped_img.save(output_path)

        print(f"{file} cropped successfully.")


input_folder = "Data\\Test"
output_folder = "Data\\ImagesNoBorders"

# Number of pixels to crop from each side
pixels_to_crop = 5

crop_images(input_folder, output_folder, pixels_to_crop)

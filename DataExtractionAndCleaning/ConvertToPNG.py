from PIL import Image
import os

def convert_to_png(input_folder, output_folder):
    for folder in os.listdir(input_folder):
        for filename in os.listdir(os.path.join(input_folder, folder)):
            if filename.endswith(".jpeg") or filename.endswith(".jpg"):
                # Open image
                with Image.open(os.path.join(input_folder, folder, filename)) as img:
                    # Convert to PNG
                    png_path = os.path.join(output_folder, folder, os.path.splitext(filename)[0] + ".png")
                    img.save(png_path, "PNG")
                    print(f"Converted {filename} to {png_path}")
# Specify input and output folders
input_folder = "WindTurbineImagesCategorization\\Data\\Dataset"
output_folder = "WindTurbineImagesCategorization\\Data\\Test"

# Convert images
convert_to_png(input_folder, output_folder)

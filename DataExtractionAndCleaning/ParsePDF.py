import fitz
import io 
import os
import PIL.Image
from math import sqrt
import re 
from pdfminer.high_level import extract_text

target_colors = [(255, 245, 0), (255, 120, 249), (255, 3, 0), (4, 0, 0), (255, 161, 0)] # Yellow, Purple, Red, Black, Orange
folder_path = "Data\\MaintenanceReports"

# Function to calculate color similarity
def color_similarity(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    # Euclidean distance between two colors
    return sqrt((r1-r2)**2+(g1-g2)**2+(b1-b2)**2)

def parseTextAndGetValues(text, flag):
    IDs = re.findall(r"unique id\s*: (\d+\s\w+)", text)
    categories = re.findall(r"Category (\w+)", text)
    
    print("IDs:", IDs)
    print("Categories:", categories)

    if flag and len(IDs) == 2 and len(categories) == 2:
        return IDs[1], categories[1]
    elif flag: # If there is only one image in the page
        return IDs[0], categories[0]
    else:
        return IDs[0], categories[0]

def get_pdf_names(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    pdfs = [folder_path + "\\" + file for file in files if file.endswith('.pdf')]
    return pdfs



pdf_names = get_pdf_names(folder_path)

counter = 1
for pdf_name in pdf_names:
    pdf = fitz.open(pdf_name)
    print(f"\n\tProcessing {pdf_name}...\n\n")

    for i in range(len(pdf)):
        page = pdf[i]
        images = page.get_images() # type: ignore
        firstImage = True # Flag to know if it is the first image in the page

        for image in images:
            base_image = pdf.extract_image(image[0])
            image_bytes = base_image["image"]
            image = PIL.Image.open(io.BytesIO(image_bytes))
            
            # Check if the image is square
            width, height = image.size
            if width != height:
                print("Image is not square. Skipping...")
                continue
            
            # Get the top-left pixel color
            top_left_pixel = image.getpixel((0, 0))

            # Check if top_left_pixel is a tuple with 3 values
            if isinstance(top_left_pixel, tuple) and len(top_left_pixel) == 3:
                
                # Check similarity with target colors
                for target_color in target_colors:
                    if color_similarity(top_left_pixel, target_color) < 50: # Tolerance value
                        print(f"\n\tTop-left pixel has a color similar to a target color, saving image at page {i+1} in {pdf_name}...")
                        # Extract text from the page
                        text = extract_text(pdf_name, page_numbers=[i])
                        ID, category = parseTextAndGetValues(text, firstImage)

                        if firstImage:
                            firstImage = False
                        
                        extension = base_image["ext"]
                        
                        image.save(open(f"Data\\Images\\img{counter}_{ID}_{category}.{extension}", "wb"))
                        print("Image saved successfully!\n\n")
                        counter += 1
            else:
                print("Top-left pixel does not contain a tuple with 3 values. Skipping...")


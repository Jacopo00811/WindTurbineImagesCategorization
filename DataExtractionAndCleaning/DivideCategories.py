import os
import shutil
import pandas as pd

def create_category_subfolders():
    categories = ["0", "1", "2", "3", "4", "5"]

    # Create subfolders for each category
    for category in categories:
        os.makedirs(os.path.join('Data\\Try', category), exist_ok=True)



excel_file_path = "DataExtractionAndCleaning\\infos-2022-05-24_new_categories.xlsx"
df = pd.read_excel(excel_file_path)

for image_name in os.listdir('Data\\ImagesNoBorders'):
    # Check if the image name corresponds to 'Unique id' in DataFrame
    image_id = image_name.split('_')[0] + "_" + image_name.split('_')[1]
    if image_id in df['Unique id'].values:
        # Get the category for the image
        category = df.loc[df['Unique id'] == image_id, 'Defect aero severity'].values[0]
        for subfolder in os.listdir('Data\\Try'):
            if category == int(subfolder):
                #print(f"Moving {"Test\\" + image_name} {category} to {"Try\\" + subfolder + "\\" + image_name}")
                shutil.move("Data\\ImagesNoBorders\\" + image_name, "Data\\Try\\" + subfolder + "\\" + image_name)
    else:
        print(f"No information found for image {image_name}")

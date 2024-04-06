import pandas as pd
import os

# Read the Excel file into a pandas DataFrame
excel_file = "DataExtractionAndCleaning\\infos-2022-05-24_new_categories.xlsx"
df = pd.read_excel(excel_file)
folder_path = "Data\\Test"

# Filter the DataFrame for rows where the "Defect location" column has "Leading Edge"
# leading_edge_defects = df[df['Defect location'] == 'Leading Edge']

# Filter the leading_edge_defects DataFrame for rows where the "Defect type" column has the specified values
specified_defects = df[df['Defect type'].isin(['Vortex Generators','Lightning strike damage', 'Uncategorized', 'Contamination', 'Dino Tails', 'LPS', 'Bearing Covers', 'Dino Shells', 'TE Delamination', 'Gurney Flaps'])]

# Extract the "Unique id" column values and save them in a list
unique_ids = specified_defects['Unique id'].tolist()

#print(len(unique_ids))

for unique_id in unique_ids:
    # Check if any image file matches the Unique id in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith(unique_id):
            # Remove the image file
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f'Removed image for Unique id: {unique_id} (File: {file_name})')
            break  # Once the file is removed, break out of the inner loop

    else:
        print(f'Image for Unique id {unique_id} not found in the folder')



specified_locations = df[df['Defect location'].isin(['Pressure Side', 'Suction Side', 'Trailing Edge'])]
unique_ids_locations = specified_locations['Unique id'].tolist()


for unique_id_location in unique_ids_locations:
    # Check if any image file matches the Unique id in the folder
    for file_name in os.listdir(folder_path):
        if file_name.startswith(unique_id_location):
            # Remove the image file
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f'Removed image for Unique id: {unique_id_location} (File: {file_name})')
            break  # Once the file is removed, break out of the inner loop

    else:
        print(f'Image for Unique id {unique_id_location} not found in the folder')
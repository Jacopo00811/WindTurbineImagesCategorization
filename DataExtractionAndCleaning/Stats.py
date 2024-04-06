import pandas as pd
import os

excel_file_path = "DataExtractionAndCleaning\\infos-2022-05-24_new_categories.xlsx"
df = pd.read_excel(excel_file_path)

leading_edge_df = df[df['Defect location'] == 'Leading Edge']

defect_types = [
    'Chipped / Peeling Paint',
    'Erosion',
    'Gouges & Scratches',
    'Major cracks',
    'Pin Holes & Voids',
    'Superficial cracks',
    'Surface Delamination'
]



# IDs in the excel file
defect_ids = {}
for defect_type in defect_types:
    defect_ids[defect_type] = leading_edge_df[leading_edge_df['Defect type'] == defect_type]['Unique id'].tolist()
countl = 0
# Print or use the lists of IDs for each defect type
for defect_type, ids in defect_ids.items():
    print(f"IDs for {defect_type}: {len(ids)}")
    countl += len(ids)
print("Total in excel: ", countl)



# Remove the damage category from the names of the images, just keep id
folder_path = "Data\\Test"
res = []
image_ids_in_folder = set()
for file_name in os.listdir(folder_path):
    image_id = '_'.join(file_name.split('_')[:2])
    image_ids_in_folder.add(image_id)

print("\nImage IDs in the folder:", len(image_ids_in_folder))
# Find IDs in defect_ids but not in the folder
for defect_type, defect_id_list in defect_ids.items():
    for unique_id in defect_id_list:
        if unique_id not in image_ids_in_folder:
            res.append(unique_id)

print("IDs in defect_ids but not in the folder:", len(res), "\n")
#print(res)
# with open('missing_IDs.txt', 'w') as file:
#     for i in res:
#     # Write 'i' into a file as a CSV
#         file.write(f'{i}\n')


# IDs in the folder without the ones in the excel file
defect_ids_in_folder = {}
for defect_type in defect_types:
    defect_ids_in_folder[defect_type] = [unique_id for unique_id in leading_edge_df[leading_edge_df['Defect type'] == defect_type]['Unique id'].tolist() if unique_id not in res]

c = 0
for defect_type, ids in defect_ids_in_folder.items():
    print(f"IDs for {defect_type}: {len(ids)}")
    c += len(ids)
print("Total in folder: ", c)


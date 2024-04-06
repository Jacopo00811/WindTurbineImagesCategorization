import os

def find_duplicate_ids(folder_path):
    id_counts = {}

    for filename in os.listdir(folder_path):
        # Split the filename into parts using '_' as delimiter
        parts = filename.split('_')
        if len(parts) < 2:
            print("Skipping:", filename)
            continue
        
        # Extract the ID part
        ID = parts[0] + '_' + parts[1]  # Combine the first two parts with an underscore
        id_counts[ID] = id_counts.get(ID, 0) + 1

    # Find duplicate IDs
    duplicate_ids = [ID for ID, count in id_counts.items() if count > 1]

    return duplicate_ids


folder_path = "Data\\Test"
duplicate_ids = find_duplicate_ids(folder_path)

if duplicate_ids:
    print("Duplicate IDs found:")
    for ID in duplicate_ids:
        print(ID)
else:
    print("No duplicate IDs found")


# Sample: 20150715_GDXV_III.jpeg
    

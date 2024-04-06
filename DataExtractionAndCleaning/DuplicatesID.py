import os

def find_duplicate_ids(folder_path):
    files = []
    duplicate = []
    # Create a dictionary where keys are strings after the first "_" and values are the count of their occurrences
    string_counts = {}
    
    for string in os.listdir(folder_path):
        # Split the string on "_", and consider only the part after the first "_"
        key = string.split("_", 1)[-1]
    #    Update the count for the key
        string_counts[key] = string_counts.get(key, 0) + 1

    return string_counts


folder_path = "Data\\Test"
    
dict = find_duplicate_ids(folder_path)
for key, value in dict.items():
    if value > 1:
        print(key)
    else:
        print("No duplicates found")

print("Done")
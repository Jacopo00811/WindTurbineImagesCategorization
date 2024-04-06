import os
import matplotlib.pyplot as plt

def plot_histogram(folder_path, show_percentage=True):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    image_counts = []

    for subfolder in subfolders:
        image_count = len([f for f in os.listdir(subfolder)])
        image_counts.append(image_count)

    total_images = sum(image_counts)
    percentages = [count / total_images * 100 for count in image_counts]

    bars = plt.bar(range(len(subfolders)), image_counts, tick_label=[f.split('\\')[-1] for f in subfolders], color='tan')

    for bar, count, percentage in zip(bars, image_counts, percentages):
        label = f'{count}'
        if show_percentage:
            label += f' ({percentage:.2f}%)'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), label, ha='center', va='bottom')
    
    plt.xlabel('Subfolders', fontweight='bold')
    plt.ylabel('Number of Images', fontweight='bold')
    plt.title(f'Histogram of Image Counts, total of: {total_images} images', fontweight='bold')
    plt.show()


folder_path = 'Data\\Pippo'
plot_histogram(folder_path, show_percentage=False)

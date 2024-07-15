from google.colab import drive
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

# Function to split and save the dataset
def split_and_save_dataset(dataset_path, train_path, test_path, test_size=0.2, random_seed=42):
    images = []
    labels = []
    classes = os.listdir(dataset_path)

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            if is_image_file(img_name):  # Ensure only image files are included
                images.append(os.path.join(class_dir, img_name))
                labels.append(idx)

    train_idx, test_idx = train_test_split(range(len(images)), test_size=test_size, stratify=labels, random_state=random_seed)

    # Function to copy files
    def copy_files(indices, destination):
        for idx in tqdm(indices):
            img_path = images[idx]
            class_name = classes[labels[idx]]
            dest_dir = os.path.join(destination, class_name)
            shutil.copy(img_path, dest_dir)

    # Copy train and test files
    copy_files(train_idx, train_path)
    copy_files(test_idx, test_path)

if __name__ == '__main__':

    # https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset

    drive.mount('/content/drive')
    dataset_path = '/content/drive/My Drive/dataset/yoga_pose'
    train_path = '/content/drive/My Drive/dataset/yoga_pose/train'
    test_path = '/content/drive/My Drive/dataset/yoga_pose/test'

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Create subdirectories for each class
    classes = os.listdir(dataset_path)
    for class_name in classes:
        os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Call the function
    split_and_save_dataset(dataset_path, train_path, test_path, test_size=0.2, random_seed=42)

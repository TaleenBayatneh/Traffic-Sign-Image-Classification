import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images(folder_path: str, image_size=(32, 32)):
    """
    Load all images from sub-folders (one folder = one class).

    Returns:
        X : np.ndarray  -> shape (n_samples, image_size*image_size*3)
        y : np.ndarray  -> class indices aligned with X
        label_map : dict -> {class_name: index}
    """
    X, y = [], []
    label_map = {}
    current_label = 0

    for class_name in sorted(os.listdir(folder_path)):
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        label_map[class_name] = current_label
        for img_file in os.listdir(class_dir):
            try:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path).resize(image_size).convert("RGB")
                X.append(np.array(img).flatten())
                y.append(current_label)
            except Exception:
                #ignore bad images
                continue
        current_label += 1

    return np.array(X), np.array(y), label_map

def prepare_dataset(folder_path: str, image_size=(32, 32), test_size=0.2):
    """
    loading data and divide it into training/testing folder using 80% , 20% 
    """
    X, y, label_map = load_images(folder_path, image_size=image_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, label_map

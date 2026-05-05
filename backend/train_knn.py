import os
import numpy as np
from skimage import io
import skimage.transform
import skimage.color
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pickle

print("Initializing Training Pipeline...")

# Configuration
DATA_PATH = "dataset"  # Your new folder with 500 images
IMG_HEIGHT = 20
IMG_WIDTH = 20
CATEGORIES = ['Negative', 'Positive'] # 0: Negative, 1: Positive

data = []
labels = []

# 1. Load and Preprocess Images
print("Loading and preprocessing images...")
for category in CATEGORIES:
    path = os.path.join(DATA_PATH, category)
    class_num = CATEGORIES.index(category)
    
    # Check if directory exists
    if not os.path.exists(path):
        print(f"Error: Directory '{path}' not found.")
        exit()

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = io.imread(img_path)
            
            # Convert to grayscale and resize
            gray_image = skimage.color.rgb2gray(img)
            resized = skimage.transform.resize(gray_image, (IMG_HEIGHT, IMG_WIDTH))
            
            # Flatten the image for KNN
            flattened_img = np.asarray(resized).ravel()
            
            data.append(flattened_img)
            labels.append(class_num)
        except Exception as e:
            pass # Skip problematic files

data = np.array(data)
labels = np.array(labels)

print(f"Successfully loaded {len(data)} images.")

# 2. Split the Data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(f"Training set size: {len(x_train)}")
print(f"Testing set size: {len(x_test)}")

# 3. Train the KNN Model
print("Training the K-Nearest Neighbors model...")
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)

# 4. Evaluate the Model
print("Evaluating model...")
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# 5. Save the Model
MODEL_FILENAME = 'concrete_knn_model.pkl'
pickle.dump(knn, open(MODEL_FILENAME, 'wb'))
print(f"Success! Model saved locally as '{MODEL_FILENAME}'")
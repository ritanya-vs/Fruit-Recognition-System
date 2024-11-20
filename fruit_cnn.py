import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from tqdm import tqdm
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


IMAGE_SIZE = 160 
BATCH_SIZE = 32 
EPOCHS = 50 
LEARNING_RATE = 1e-3 


root_dir = "C:/Users/nivit/Downloads/archive (3)/Fruits Classification/"
train_dir = f"{root_dir}train/" 
valid_dir = f"{root_dir}valid/"
test_dir = f"{root_dir}test/"
LOSS = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

class_names = sorted(os.listdir(train_dir))
n_classes = len(class_names)
print(f"Total number of classes: {n_classes}")
print(f"Classes: {class_names}")

def load_image(image_path: str) -> tf.Tensor:
    assert os.path.exists(image_path), f'Invalid image path: {image_path}'
    image = tf.io.read_file(image_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except:
        image = tf.image.decode_png(image, channels=3)
    image = tf.image.conver
    
    t_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return tf.cast(image, tf.float32)

def load_dataset(root_path: str, class_names: list, trim: int=None) -> Tuple[np.ndarray, np.ndarray]:
    if trim:
        n_samples = len(class_names) * trim
    else:
        n_samples = sum([len(os.listdir(os.path.join(root_path, name))) for name in class_names])

    images = np.empty(shape=(n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    labels = np.empty(shape=(n_samples, 1), dtype=np.int32)

    n_image = 0
    for class_name in tqdm(class_names, desc="Loading"):
        class_path = os.path.join(root_path, class_name)
        image_paths = list(glob(os.path.join(class_path, "*")))[:trim]
        for file_path in image_paths:
            image = load_image(file_path)
            label = class_names.index(class_name)
            images[n_image] = image
            labels[n_image] = label
            n_image += 1

    indices = np.random.permutation(n_samples)
    return images[indices], labels[indices]


X_train, y_train = load_dataset(root_path=train_dir, class_names=class_names, trim=1000)
X_valid, y_valid = load_dataset(root_path=valid_dir, class_names=class_names)
X_test, y_test = load_dataset(root_path=test_dir, class_names=class_names)


def show_images(images: np.ndarray, labels: np.ndarray, n_rows: int=1, n_cols: int=5, figsize: tuple=(25, 8)):
    for row in range(n_rows):
        plt.figure(figsize=figsize)
        rand_indices = np.random.choice(len(images), size=n_cols, replace=False)
        for col, index in enumerate(rand_indices):
            image = images[index]
            label = class_names[int(labels[index])]
            plt.subplot(1, n_cols, col+1)
            plt.imshow(image)
            plt.title(label.title())
            plt.axis("off")
        plt.show()

show_images(images=X_train, labels=y_train, n_rows=2)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])


cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=LOSS,
    metrics=METRICS
)


cnn_model.summary()


history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("cnn_fruit_classification.keras", save_best_only=True)
    ]
)



y_pred = np.argmax(cnn_model.predict(X_test), axis=-1)
print(classification_report(y_test, y_pred, target_names=class_names))


converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_path = "cnn_fruit_classification.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {tflite_model_path}")

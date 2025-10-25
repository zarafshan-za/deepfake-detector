!pip install Kaggle
import os
import zipfile
!pip install tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d shivamardeshna/real-and-fake-images-dataset-for-image-forensics
with zipfile.ZipFile("real-and-fake-images-dataset-for-image-forensics.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

import os

root = "dataset"
for i in range(1, 5):
    dataset_path = os.path.join(root, f"Data Set {i}", f"Data Set {i}")
    print(f"\nContents of: {dataset_path}")

    for split in ["train", "test", "validation"]:
        for category in ["real", "fake"]:
            folder = os.path.join(dataset_path, split, category)
            count = len(os.listdir(folder))
            print(f"{split}/{category}: {count} images")

!pip install tensorflow
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model

base_dir = "dataset/Data Set 1/Data Set 1"

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

img_height, img_width = 299, 299  # Required for Xception
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1

)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

import numpy as np

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = Dropout(0.6)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with label smoothing

loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn, metrics=['accuracy'])

# Define callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

for layer in base_model.layers[-10:]:
    layer.trainable = True

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

from keras.saving import save_model
save_model(model, "deepfake_detector_model.keras")

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

from google.colab import drive
from tensorflow.keras.models import load_model

drive.mount('/content/drive')
model1_path = '/content/drive/MyDrive/Models/deepfake_detector_model_testacc_77.keras'
model1 = load_model(model1_path)
test_loss, test_acc = model1.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

model.save('deepfake_detector_model.keras')

from google.colab import files
files.download('deepfake_detector_model.keras')

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict
y_pred_probs = model3.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype("int32").flatten()
y_true = test_generator.classes

# Classification report
report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'], output_dict=True)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))

# Bar chart for metrics
metrics = ['precision', 'recall', 'f1-score']
values_fake = [report['Fake'][m] for m in metrics]
values_real = [report['Real'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(x - width/2, values_fake, width, label='Fake', color='salmon')
plt.bar(x + width/2, values_real, width, label='Real', color='skyblue')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('Evaluation Metrics')
plt.xticks(ticks=x, labels=metrics)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()


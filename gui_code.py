import sys
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QFileDialog, QHBoxLayout, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np

# Load the model once
model = tf.keras.models.load_model("model.keras")
IMG_SIZE = 299

class DeepfakeDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Deepfake Image Detector")
        self.setGeometry(200, 200, 600, 700)
        self.is_dark_mode = False
        self.initUI()

    def initUI(self):
        font_large = QFont("Segoe UI", 14)
        font_medium = QFont("Segoe UI", 11)

        self.image_label = QLabel("Upload an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; color: black;")
        self.image_label.setFixedSize(400, 400)
        self.image_label.setFont(font_medium)

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(font_large)
        self.result_label.setStyleSheet("color: black;")

        self.upload_button = QPushButton("📤 Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFont(font_medium)

        self.predict_button = QPushButton("🔍 Predict")
        self.predict_button.clicked.connect(self.predict_image)
        self.predict_button.setFont(font_medium)
        self.predict_button.setEnabled(False)

        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setFont(font_medium)

        self.theme_toggle_button = QPushButton("🌓 Toggle Dark/Light Mode")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        self.theme_toggle_button.setFont(font_medium)

        hbox = QHBoxLayout()
        hbox.addWidget(self.upload_button)
        hbox.addWidget(self.predict_button)
        hbox.addWidget(self.reset_button)

        vbox = QVBoxLayout()
        vbox.setSpacing(20)
        vbox.setAlignment(Qt.AlignCenter)
        vbox.addSpacerItem(QSpacerItem(0, 20))
        vbox.addWidget(self.image_label, alignment=Qt.AlignCenter)
        vbox.addWidget(self.result_label)
        vbox.addLayout(hbox)
        vbox.addWidget(self.theme_toggle_button, alignment=Qt.AlignCenter)

        self.setLayout(vbox)
        self.apply_light_theme()

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setStyleSheet("")  # Remove placeholder border
            self.result_label.setText("")
            self.predict_button.setEnabled(True)

    def predict_image(self):
        img = Image.open(self.image_path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        percent = int(round((1 - prediction) * 100)) if prediction < 0.5 else int(round(prediction * 100))
        if prediction >= 0.5:
            result = f"❌ Fake ({percent}%)"
        else:
            result = f"✅ Real ({percent}%)"
        self.result_label.setText(f"Prediction: {result}")

    def reset(self):
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Upload an image")
        self.image_label.setStyleSheet("border: 2px dashed #aaa;")
        self.result_label.setText("")
        self.predict_button.setEnabled(False)

    def toggle_theme(self):
        if self.is_dark_mode:
            self.apply_light_theme()
        else:
            self.apply_dark_theme()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #444;
                color: white;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.image_label.setStyleSheet("border: 2px dashed #888; color: white;")
        self.result_label.setStyleSheet("color: white;")
        self.is_dark_mode = True

    def apply_light_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                color: black;
            }
            QLabel {
                color: black;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: black;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; color: black;")
        self.result_label.setStyleSheet("color: black;")
        self.is_dark_mode = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepfakeDetectorGUI()
    window.show()
    sys.exit(app.exec_())

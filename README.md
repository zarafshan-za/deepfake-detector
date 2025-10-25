# Real vs. Fake Image Detector
A university project made for a course (Digital Image Processing). This project involves building a deep learning-based image classifier that can detect whether an image is real or deepfake (with 80% accuracy). It utilizes a pretrained CNN model finetuned on a deepfake dataset and includes a modern PyQt-based GUI for user interaction.
# Python Dependencies
• Python Version: 3.10
• Tensorflow
• Pillow
• PyQt5
• Numpy
# Dataset Used
https://www.kaggle.com/datasets/shivamardeshna/real-and-fake-images-dataset-for-imageforensics
# Installation & Usage:
1. Download trained model file here
[model.keras]([link](https://drive.google.com/file/d/1MiNVqAB9vqECZJltewyikJ0toplLrycI/view?usp=sharing))
1. Clone this repository:
```bash
git clone https://github.com/zarafshan-za/deepfake-detector.git
```
2. Enter the project folder
```bash
cd deepfake-detector
```
4. Install dependencies
```bash
pip install Pillow==8.4.0
```
```bash
pip install tensorflow pyqt5 numpy
```
6. Launch the application
```bash
py -3.10 gui_code.py
```

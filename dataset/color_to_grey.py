from PIL import Image
import os
root_dir = r'F:\kaggle_eye\dataset'
file_list = os.listdir(r'F:\kaggle_eye\dataset\test')
for file in file_list:
    image_file = Image.open(os.path.join(r'F:\kaggle_eye\dataset\test', file))  # open colour image
    image_file = image_file.convert('L')  # convert image to black and white
    image_file.save(os.path.join(r'F:\kaggle_eye\dataset', file))





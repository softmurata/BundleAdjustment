import numpy as np
from PIL import Image

def load_any_img(file_path):
	img = Image.open(file_path)
	return img

def load_img(file_path):
	img = Image.open(file_path).convert('RGB')
	return img

def get_sin_cos(theta):
	return np.array([[np.sin(theta)], [np.cos(theta)]])

import argparse
import numpy as np
import tensorflow as tf
from utils import *
from keras.preprocessing import image
import cv2
import scipy
import scipy.stats

parser = argparse.ArgumentParser(description='Training Shared Feature Extractor')
parser.add_argument("input_image", nargs = 1)
parser.add_argument('--model_filename', type=str, default = './models/anonymous_2_bd_net.h5', help='Path to the model')
parser.add_argument('--validation_data', type=str, default = './data/clean_validation_data.h5', help='Path to the validation data')
parser.add_argument('--percent', default=0.01, type=float, help='Percent of FAR for choosing threshold')
parser.add_argument('--random', action='store_true', default=False, help='Activates the mode to superimpose random perturbations (Default: False)')
args = parser.parse_args()

image_path = args.input_image[0]
input_ = cv2.imread(image_path)
input_ = data_preprocess(input_.astype('float32'))

x_valid, y_valid = data_loader(args.validation_data)
x_valid = data_preprocess(x_valid)
n_classes = len(np.unique(y_valid))

model = tf.keras.models.load_model(args.model_filename)

N = 100

if args.random:
   H_list = np.load('entropy_lists/H_list_rnd_2.npy')
else:
   H_list = np.load('entropy_lists/H_list_2.npy')
threshold = scipy.stats.norm.ppf(args.percent, loc=np.array(H_list).mean(), scale=np.array(H_list).std())

if args.random:
   output = classify_sample_rand(model, input_, threshold, N, n_classes)
else:
   output = classify_sample(model, input_, x_valid, threshold, N, n_classes)
print(output)

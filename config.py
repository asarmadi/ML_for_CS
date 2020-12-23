import argparse
import numpy as np
import tensorflow as tf
from utils import *
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
import scipy
import scipy.stats

parser = argparse.ArgumentParser(description='Training Shared Feature Extractor')
parser.add_argument('--model_filename', type=str, default = './models/multi_trigger_multi_target_bd_net.h5', help='Path to the model')
parser.add_argument('--validation_data', type=str, default = './data/clean_validation_data.h5', help='Path to the validation data')
parser.add_argument('--best_N', action='store_true', default=False, help='Activates the mode to find the best N (Default: False)')
parser.add_argument('--random', action='store_true', default=False, help='Activates the mode to superimpose random perturbations (Default: False)')
args = parser.parse_args()

max_N = 20

x_valid, y_valid = data_loader(args.validation_data)
x_valid = data_preprocess(x_valid)
n_classes = len(np.unique(y_valid))

model = tf.keras.models.load_model(args.model_filename)

str_name = ''
if args.random:
   str_name = 'rnd'

if args.best_N:
   std_list = []
   for n in range(1,max_N):
       if args.random:
          H_list = find_entropy_list(model, x_valid, x_valid, n)
       else:
          H_list = find_entropy_list_rand(model, x_valid, n)
       std_list.append(np.array(H_list).std())

   np.save('std_list_'+str_name+'.npy', std_list)

   plt.figure()
   plt.scatter(np.arange(1,max_N),std_list)
   plt.xlabel('N')
   plt.ylabel('Standard Deviation')
   plt.savefig('Figs/std_vs_N_'+str_name+'.png')
   plt.close()

N = 100

if args.random:
   H_list = find_entropy_list_rand(model, x_valid, N)
else:
   H_list = find_entropy_list(model, x_valid, x_valid, N)

np.save('entropy_lists/H_list.npy', H_list)

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
parser.add_argument('--model_filename', type=str, default = './models/sunglasses_bd_net.h5', help='Path to the model')
parser.add_argument('--validation_data', type=str, default = './data/clean_validation_data.h5', help='Path to the validation data')
parser.add_argument('--test_data', type=str, default = './data/clean_test_data.h5', help='Path to the test data')
parser.add_argument('--percent', default=0.01, type=float, help='Percent of FAR for choosing threshold')
parser.add_argument('--best_N', action='store_true', default=False, help='Activates the mode to find the best N')
args = parser.parse_args()

max_N = 20

x_valid, y_valid = data_loader(args.validation_data)
x_valid = data_preprocess(x_valid)
n_classes = len(np.unique(y_valid))

x_test, y_test = data_loader(args.test_data)
x_test = data_preprocess(x_test)

model = tf.keras.models.load_model(args.model_filename)

if args.best_N:
   std_list = []
   for n in range(1,max_N):
       H_list = find_entropy_list(model, x_valid, x_valid, n)
       std_list.append(np.array(H_list).std())

   np.save('std_list.npy', std_list)

   plt.figure()
   plt.scatter(np.arange(1,max_N),std_list)
   plt.xlabel('N')
   plt.ylabel('Standard Deviation')
   plt.savefig('Figs/std_vs_N_rand.png')
   plt.close()

   std_list = np.load('std_list.npy')
   N = 1
   for i in range(1,len(std_list)-1):
       if (std_list[i] - std_list[i+1])<0.01:
          N = i
else:
   N = 12
   
H_list = find_entropy_list(model, x_valid, x_valid, N)

threshold = scipy.stats.norm.ppf(args.percent, loc=np.array(H_list).mean(), scale=np.array(H_list).std())

FN = 0
tot = 0
for i, input_ in enumerate(x_test):
    output = classify_sample(model, input_, x_valid, threshold, N, n_classes)
    if output!=y_test[i]:
       FN += 1
    tot += 1

print('FN: {} {}/{}'.format(FN/tot,FN,tot))

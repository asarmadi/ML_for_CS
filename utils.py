import cv2
import h5py
import numpy as np

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def save_image(img, name):
    img = image.array_to_img(K.eval(img[0]))
    img.save('Figs/'+name+'.png', 'png')

def cal_entropy(pi):
    pi = np.ma.masked_equal(pi,0)
    H = pi*np.log2(pi)
    return -1*H.sum()

def find_entropy_list(model, x_test, x_valid, N):
    H_list = []
    all_indices = np.arange(len(x_valid))
    for i, input_ in enumerate(x_test):
        subset_indices = np.random.choice(all_indices, size=N)
        H = 0
        for j in subset_indices:
            perturbed_input = cv2.addWeighted(input_,1,x_valid[j],1,0,dtype=cv2.CV_64F)
            output = model(np.expand_dims(perturbed_input,axis=0))
            H += cal_entropy(output)
        H /= N
        H_list.append(H)
    return H_list

def classify_sample(model, sample, x_valid, threshold, N, n_classes):
    all_indices = np.arange(len(x_valid))
    subset_indices = np.random.choice(all_indices, size=N)
    H = 0
    for j in subset_indices:
        perturbed_input = cv2.addWeighted(sample,1,x_valid[j],1,0,dtype=cv2.CV_64F)
        output = model(np.expand_dims(perturbed_input,axis=0))
        H += cal_entropy(output)
    H /= N
    if H >= threshold:
       return np.argmax(model(np.expand_dims(sample,axis=0)), axis=1)
    else:
       return n_classes

def find_entropy_list_rand(model, x_test, N):
    H_list = []
    for i, input_ in enumerate(x_test):
        H = 0
        for j in range(N):
            perturbed_input = cv2.addWeighted(input_,1,np.random.rand(input_.shape[0],input_.shape[1],input_.shape[2]),1,0,dtype=cv2.CV_64F)
            output = model(np.expand_dims(perturbed_input,axis=0))
            H += cal_entropy(output)
        H /= N
        H_list.append(H)
    return H_list

def classify_sample_rand(model, sample, threshold, N, n_classes):
    H = 0
    for j in range(N):
        perturbed_input = cv2.addWeighted(sample,1,np.random.rand(sample.shape[0],sample.shape[1],sample.shape[2]),1,0,dtype=cv2.CV_64F)
        output = model(np.expand_dims(perturbed_input,axis=0))
        H += cal_entropy(output)
    H /= N
    if H >= threshold:
       return np.argmax(model(np.expand_dims(sample,axis=0)), axis=1)
    else:
       return n_classes

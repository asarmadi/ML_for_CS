# ML_for_CS

I have implemented the method described in ["STRIP: A Defence Against Trojan Attacks on Deep Neural Networks"](https://dl.acm.org/doi/pdf/10.1145/3359789.3359790).

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.4.3
   3. Numpy 1.18.5
   4. Matplotlib 2.2.2
   5. H5py 2.10.0
   6. TensorFlow-gpu 2.3.1
   7. Opencv-python 4.4.0.46

## II. Evaluating the Method 
   1. To evaluate the STRIP method, execute `strip.py` by running:  
      `python strip.py --model_filename model_dir --validation_data val_dir --test_data test_dir --percent pct --best_N`.
      in which `model_dir` is the path to the model, `val_dir` is the path to the file containing clean validation data, `test_dir` is the path to the file containing samples to be evaluated, `pct` is a float number between 0 and 1 that determines the percentage of FAR for choosing threshold, and the argument `best_N` can be added to the above command to specify that the best N should be calculated.

## III. Results
   1. When the `best_N` argument is added, a figure will be generated under Figs folder which depicts the variance of entropies with respect to the different values of `N`. The maximum value of `N` is set to be 20 in the code.
   ![Best N](/Figs/std_vs_N.png)

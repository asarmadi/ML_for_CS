# ML_for_CS

The STRIP method ["STRIP: A Defence Against Trojan Attacks on Deep Neural Networks"](https://dl.acm.org/doi/pdf/10.1145/3359789.3359790) has been implemented in this repository. Also, another version of STRIP has been developed in which instead of superimposing the images with other images from the validation, random perturbations are added to the samples.

## I. Dependencies
   1. Python 3.6.9
   2. Keras 2.4.3
   3. Numpy 1.18.5
   4. Matplotlib 2.2.2
   5. H5py 2.10.0
   6. TensorFlow-gpu 2.3.1
   7. Opencv-python 4.4.0.46

## II. Initialization 
**(Note: This step could be ignored if you want to evaluate the method, since corresponding entropy lists are generated under entropy_lists folder)**
To generate entropy lists for the vailable validation data, execute `config.py` by running:  
      `python config.py --model_filename model_dir --validation_data val_dir --percent pct --best_N --random`.
      in which `model_dir` is the path to the model, `val_dir` is the path to the file containing clean validation data, `pct` is a float number between 0 and 1 that determines the percentage of FAR for choosing threshold, the argument `best_N` can be added to the above command to specify that the best N should be calculated, and the argument `random` will determine whether the random perturbation of the samples should be used for perturbing the inputs or not.
      
## III. Evaluating the Method
There are 4 files for evaluation of the STRIP on 4 different models named: eval1.py, eval2.py, eval3.py, and eval4.py. Following are the examples for running each file:

      `python eval1.py test_img.png`
      
      `python eval2.py test_img.png`
      
      `python eval3.py test_img.png`
      
      `python eval4.py test_img.png`
      
      in which `test_img.png` is the input image to be evaluated. The output range is [0,1283] and 1283 corresponds to a poisoned image.
## IV. Results
   1. When the `best_N` argument is added, a figure will be generated under Figs folder which depicts the variance of entropies with respect to the different values of `N`. The maximum value of `N` is set to be 20 in the code. Following figure is a sample output of the method.
   ![Best N](/Figs/std_vs_N.png)

   2. Then the threshold will be calculated based on the `N`, `pct`, and validation data.
   3. The False negative rate of the method on the test data will be reported.

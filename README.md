Handwritten digit predictor using simple neural networks

Dataset used - MNIST dataset (imported directly using keras)
Dataset info : Images of digits were taken from a variety of scanned documents, normalized in size and centered. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required.

Each image is a 28 by 28 pixel square (784 pixels total). A standard spit of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error, which is nothing more than the inverted classification accuracy.


Implemented entirely in keras using tensorflow(CPU) as the backend.

A sequential model is used to implement the neural networks here.
 
A baseline error of around only 1 to 2% is generated.

The training set is saved in a .json file and a .h5 file(weights).
Training sets consists of data trained in 10 epochs. My CPU  took around 60 s to process the training set and return the baseline error

Source code - src.py


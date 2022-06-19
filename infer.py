import numpy as np
import tensorflow as tf
import cv2

# Infer tflite model in model
# To infer edgetpu model in c++ see `infer` dir.

image = cv2.imread("infer_cpp/example2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/mnist.tflite")

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = cv2.resize(gray, list(input_shape)[1:])
input_data = np.expand_dims(input_data, axis=[0])


interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

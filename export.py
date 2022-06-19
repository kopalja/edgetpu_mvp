import os
import tensorflow as tf

# Quantization aware training - does not compile for edgetpu
model = tf.keras.models.load_model('model/trained')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.experimental_new_converter = True
quantized_tflite_model = converter.convert()

tflite_mode_path = os.path.join("model", "mnist.tflite")

with open(tflite_mode_path, "wb") as f:
    f.write(quantized_tflite_model)


os.system(f"edgetpu_compiler {tflite_mode_path} -o model")
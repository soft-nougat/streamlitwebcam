# -*- coding: utf-8 -*-
"""

Script with tf lite model related functions (all functions taken from below colab)
Model used: ssd_mobiledet_cpu_coco_int8.tflite
Reference google colab: https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/MobileDet_Conversion_TFLite.ipynb#scrollTo=_rz1wbDv58t2
Special thanks to the author of the colab - sayakpaul :)

"""
import numpy as np
import cv2
import tensorflow as tf
import re
import helper as help

def display_results(LABELS, COLORS, HEIGHT, WIDTH, image_path, interpreter, threshold):
    '''
    
    Main function to read and prepare input, draw boxes and return image
    
    Parameters
    ----------
    LABELS : Labels defined in load_labels()
    COLORS : Colors defined in define_tf_lite_model()
    HEIGHT : Image height defined in define_tf_lite_model()
    WIDTH : Image width in define_tf_lite_model()
    image_path : Where to get the image from, in this app TempDir
    interpreter : Interpreter defined in define_tf_lite_model()
    threshold : The accuracy threshold.

    Returns
    -------
    original_numpy : Image with bouding boxes and detected objects

    '''
    # Load the input image and preprocess it
    input_type = interpreter.get_input_details()[0]['dtype']
    preprocessed_image, original_image = preprocess_image(HEIGHT, WIDTH, image_path, input_type)
    
    # =============Perform inference=====================
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

    # =============Display the results====================
    original_numpy = original_image.numpy()
    counter = 0
    for obj in results:
        # set counter of text
        counter = counter + 1
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_numpy.shape[1])
        xmax = int(xmax * original_numpy.shape[1])
        ymin = int(ymin * original_numpy.shape[0])
        ymax = int(ymax * original_numpy.shape[0])

        # Grab the class index for the current iteration
        idx = int(obj['class_id'])
        # Skip the background
        if idx >= len(LABELS):
            continue

        # Draw the bounding box and label on the image
        color = [int(c) for c in COLORS[idx]]
        cv2.rectangle(original_numpy, (xmin, ymin), (xmax, ymax), 
                    color, 2)
        y = ymin - 15 if ymin - 15 > 15 else ymin + 15
        label = "{}: {:.2f}%".format(LABELS[obj['class_id']],
            obj['score'] * 100)
        cv2.putText(original_numpy, label, (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        score = obj['score'] * 100
       
        help.sub_text(str(counter) + ') The model has detected a(an): ' + 
                      LABELS[obj['class_id']] + ' with ' + 
                      str(score) + ' confidence.')

    # Return the final image
    if (input_type==np.float32) & (original_numpy.max()==1.0):
        original_numpy = (original_numpy * 255).astype(np.uint8)
    return original_numpy

def define_tf_lite_model():
    '''
    
    Function to define labels, colors, height and width of model
    Also allocates tensors
    
    '''
    # Load the labels and define a color bank
    LABELS = load_labels("final_model/coco_labels.txt")
    
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), 
                                dtype="uint8")
    
    interpreter = tf.lite.Interpreter(model_path='final_model/ssd_mobiledet_cpu_coco_int8.tflite')
    interpreter.allocate_tensors()
    
    _, HEIGHT, WIDTH, _ = interpreter.get_input_details()[0]['shape']
    
    return(LABELS, COLORS, HEIGHT, WIDTH, interpreter)

def load_labels(path):
  '''
  
  Open labels from root folder
  
  ''' 
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  '''
  
  Set input tensor, call interpreter and get input details
  
  '''
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  if interpreter.get_input_details()[0]["dtype"]==np.uint8:
      input_scale, input_zero_point = interpreter.get_input_details()[0]["quantization"]
      image = image / input_scale + input_zero_point
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  '''
  
  Get the output tensor
  
  '''
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  '''
  
  Returns a list of detection results
  
  '''
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def preprocess_image(HEIGHT, WIDTH, image_path, input_type=np.float32):
    '''
    
    Reads image from file path and converts to tf readable
    
    '''
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3)
    original_image = img
    if input_type == np.uint8:
        img = tf.image.convert_image_dtype(img, tf.uint8)
    else:
        img = tf.image.convert_image_dtype(img, tf.float32)
    resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
    resized_img = resized_img[tf.newaxis, :]
    return resized_img, original_image

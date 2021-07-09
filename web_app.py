# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:39:38 2021

@author: TNIKOLIC

A streamlit app to call streamlit component webrtc and load a tf lite model for image classification
"""

import base64
import threading
from typing import Union
from PIL import Image
import av
import numpy as np
import streamlit as st
from datetime import date
import time
import cv2
import tensorflow as tf
import re
import os 
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

def main():
    set_bg_hack()
    
    
    # Main panel setup
    display_app_header(main_txt='Object detection app',
                       sub_txt='Welcome to the TF Lite Object Detection App. This app allows you to take/upload photos and classify them using a tf lite model.')
    
    option = st.selectbox(
        'Please select photo input type',
        ('None', 'Take photo', 'Upload photo'))
    
    # Load the labels and define a color bank
    LABELS = load_labels("final_model/coco_labels.txt")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), 
                                dtype="uint8")
    interpreter = tf.lite.Interpreter(model_path='final_model/ssd_mobiledet_cpu_coco_int8.tflite')
    interpreter.allocate_tensors()
    _, HEIGHT, WIDTH, _ = interpreter.get_input_details()[0]['shape']
    
    if option == 'Take photo':
   
        class VideoTransformer(VideoTransformerBase):
            frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
            in_image: Union[np.ndarray, None]
            out_image: Union[np.ndarray, None]
    
            def __init__(self) -> None:
                self.frame_lock = threading.Lock()
                self.in_image = None
                self.out_image = None
    
            def transform(self, frame: av.VideoFrame) -> np.ndarray:
                in_image = frame.to_ndarray(format="bgr24")
    
                out_image = in_image[:, ::-1, :]  # Simple flipping for example.
    
                with self.frame_lock:
                    self.in_image = in_image
                    self.out_image = out_image
    
                return out_image
        
        ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

        if ctx.video_transformer:
            if st.button("Snapshot"):
                with ctx.video_transformer.frame_lock:
                    in_image = ctx.video_transformer.in_image
                    out_image = ctx.video_transformer.out_image
    
                if in_image is not None and out_image is not None:
                    #st.write("Input image:")
                    #st.image(in_image, channels="BGR")
                    display_app_header("Output image:",
                                               "")
                    st.image(out_image, channels="BGR")
                    
                    today = date.today()
                    d = today.strftime("%b-%d-%Y")
                    
                    t = time.localtime()
                    current_time = time.strftime("%H-%M-%S", t)
                    
                    file_name = "tempDir/photo_" + d + "_" + current_time + ".jpg"
                    
                    #out_image = Image.fromarray(out_image)
                    
                    cv2.imwrite(file_name, out_image)
                       
                    display_app_header("Object detection:",
                                               "")
             
                    resultant_image = display_results(LABELS, 
                                                      COLORS, 
                                                      HEIGHT, 
                                                      WIDTH, 
                                                      file_name, 
                                                      interpreter, 
                                                      threshold=0.5)
                    
                    st.image(Image.fromarray(resultant_image), use_column_width=True)
                    
                else:
                    st.warning("No frames available yet.")
    
    elif option == 'Upload photo':
        
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg","png"])
        
        if uploaded_file is not None:
            
            st.image(uploaded_file)
            
            # https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/#comments
            
            with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())  
             
            resultant_image = display_results(LABELS, 
                                              COLORS, 
                                              HEIGHT, 
                                              WIDTH, 
                                              "tempDir/" + uploaded_file.name, 
                                              interpreter, 
                                              threshold=0.5)
            
            st.image(Image.fromarray(resultant_image), use_column_width=True)
        
    else:
        display_app_header("Please select the type of photo you would like to classify.",
                           "")



def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
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
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  if interpreter.get_input_details()[0]["dtype"]==np.uint8:
      input_scale, input_zero_point = interpreter.get_input_details()[0]["quantization"]
      image = image / input_scale + input_zero_point
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
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

#@title Inference utils
def display_results(LABELS, COLORS, HEIGHT, WIDTH, image_path, interpreter, threshold=0.1):
    # Load the input image and preprocess it
    input_type = interpreter.get_input_details()[0]['dtype']
    preprocessed_image, original_image = preprocess_image(HEIGHT, WIDTH, image_path, input_type)
    
    # =============Perform inference=====================
    start_time = time.monotonic()
    results = detect_objects(interpreter, preprocessed_image, threshold=threshold)
    print(f"Elapsed time: {(time.monotonic() - start_time)*1000} miliseconds")

    # =============Display the results====================
    original_numpy = original_image.numpy()
    for obj in results:
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
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Return the final image
    if (input_type==np.float32) & (original_numpy.max()==1.0):
        original_numpy = (original_numpy * 255).astype(np.uint8)
    return original_numpy

def set_bg_hack():
    # set bg name
    main_bg = "background.png"
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .reportview-container {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# display app header and sidebar
# use HTML code to set div
def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed 
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <h2 style = "color:#F26531; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#1F4E79; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)


if __name__ == "__main__":
    main()
        


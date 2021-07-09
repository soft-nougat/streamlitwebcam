# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:39:38 2021

@author: TNIKOLIC

A streamlit app to call streamlit component webrtc and load a tf lite model for image classification
"""

import base64
from PIL import Image
import streamlit as st
from datetime import date
import time
import cv2
import os 
import object_detection as detect
import snapshot as snap

def main():
    
    # Main panel setup
    # Set website details
    st.set_page_config(page_title ="Webcam snapshot object detection", 
                       page_icon=':camera:', 
                       layout='centered')
    
    # Set the background
    set_bg_hack()
    
    # Set app header
    text = """
    <center> <br> Welcome to the Sogeti Object Detection App. </br> </center>
    <center> <br> This app allows you to take/upload photos and 
    detect objects in them using a tf lite model. </br>
    <br> The model used is: <b>ssd_mobiledet_cpu_coco_int8.tflite</b>.
    This model was downloaded from 
    <a href= 'https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/MobileDet_Conversion_TFLite.ipynb#scrollTo=_rz1wbDv58t2'>this google colab</a>.
    Special thanks to the author of this notebook <a href = 'https://github.com/sayakpaul'> sayakpaul </a>
    </center>
    """
    
    display_app_header(main_txt='Object detection app',
                       sub_txt= text)
    
    # Set selectbox
    option = st.selectbox(
        'Please select photo input type',
        ('None', 'Take photo', 'Upload photo'))
    
    # Get all model details
    labels, colors, height, width, interpreter = detect.define_tf_lite_model()
    
    # Start with app logic:
    if option == 'Take photo':
        
        # In case Take photo is selected, run the webrtc component, 
        # save photo and pass it to the object detection model
        out_image = snap.streamlit_webrtc_snapshot()
        
        # If the image is not empty, display it and pass to model
        if out_image is not None:
            
            display_app_header("Your image:",
                               "")
            st.image(out_image, channels="BGR")
               
            display_app_header("Object detection:",
                               "")
            
            file_name = write_image(out_image)
     
            object_detection = detect.display_results(labels, 
                                                      colors, 
                                                      height, 
                                                      width,
                                                      file_name, 
                                                      interpreter, 
                                                      threshold=0.5)
            
            st.image(Image.fromarray(object_detection), 
                     use_column_width=True)
           
        # In case ICE state is not successful, show warning
        else:
            st.warning("No frames available yet.")
    
    # If option is upload photo, allow upload and pass to model
    elif option == 'Upload photo':
        
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg","png"])
        
        if uploaded_file is not None:
            
            st.image(uploaded_file)
            
            # https://blog.jcharistech.com/2021/01/21/how-to-save-uploaded-files-to-directory-in-streamlit-apps/#comments
            
            with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer())  
             
            resultant_image = detect.display_results(labels, 
                                                     colors, 
                                                     height, 
                                                     width,
                                                     "tempDir/" + uploaded_file.name, 
                                                     interpreter, 
                                                     threshold=0.5)
            
            st.image(Image.fromarray(resultant_image), use_column_width=True)
        
    else:
        display_app_header("Please select the type of photo you would like to classify.",
                           "")

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

def write_image(out_image):
    '''
    
    Write image to tempDir folder with a unique name
    
    '''
    
    today = date.today()
    d = today.strftime("%b-%d-%Y")
    
    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)
    
    file_name = "tempDir/photo_" + d + "_" + current_time + ".jpg"
    
    cv2.imwrite(file_name, out_image)
    
    return(file_name)

if __name__ == "__main__":
    main()
        


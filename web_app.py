# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:39:38 2021

@author: TNIKOLIC

A streamlit app to call streamlit component webrtc and load a tf lite model for object detection
"""

# import main packages
import streamlit as st 
from PIL import Image # PIL is used to display images 
import os # used to save images in a directory
# import script functions 
import object_detection as detect
import snapshot as snap
import helper as help

def main():
    
    # ===================== Set page config and background =======================
    # Main panel setup
    # Set website details
    st.set_page_config(page_title ="Webrtc object detection", 
                       page_icon=':camera:', 
                       layout='centered')
    
    # Set the background
    help.set_bg_hack()
    
    # ===================== Set header and site info =============================
    # Set app header
    help.header('Object detection app')
    
    # Set text and pass to sub_text function
    text = """
    <center> <br> Welcome to the Sogeti Object Detection App. </br> </center>
    <center> This app allows you to take/upload photos and detect objects in them using a tf lite model. 
    </center>
    """
    help.sub_text(text)
    
    # Set expander with references and special mentions
    help.expander()
    
    
    # ======================= Get tf lite model details ==========================
    labels, colors, height, width, interpreter = detect.define_tf_lite_model()
    
    # ============================= Main app =====================================
    option = st.selectbox(
        'Please select photo input type',
        ('None', 'Take photo', 'Upload photo'))
    
    # Start with app logic:
    if option == 'Take photo':
        
        # In case Take photo is selected, run the webrtc component, 
        # save photo and pass it to the object detection model
        out_image = snap.streamlit_webrtc_snapshot()
        
        if out_image is not None:
            
            help.header("Your photo")
            
            st.image(out_image, channels="BGR")
            
            file_name = help.write_image(out_image)
     
            object_detection = detect.display_results(labels, 
                                                      colors, 
                                                      height, 
                                                      width,
                                                      file_name, 
                                                      interpreter, 
                                                      threshold=0.5)
            st.image(Image.fromarray(object_detection), 
                     use_column_width=True)
            
        else:
            
            st.warning('Waiting for snapshot to be taken')
           
    # If option is upload photo, allow upload and pass to model
    elif option == 'Upload photo':
        
        uploaded_file = st.file_uploader("Upload a photo", type=["jpg","png"])
        
        if uploaded_file is not None:
            
            st.image(uploaded_file)
            
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
        help.header("Please select the type of photo you would like to classify.")


if __name__ == "__main__":
    main()
        


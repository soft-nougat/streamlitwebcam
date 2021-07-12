# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:05:45 2021

@author: TNIKOLIC

Helper script with functions re design and saving images
"""

import streamlit as st
import base64 # used in set_bg_hack to encode bg image
from datetime import date # used in write_image for file name
import time # used in write_image for file name
import cv2 # used in write_image for writing images

def set_bg_hack():
    '''
    A function to unpack an image from root folder and set as bg.
    The bg will be static and won't take resolution of device into account.

    Returns
    -------
    The background.

    '''
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
        
def header(text):
    '''
     A function to neatly display headers in app.

    Parameters
    ----------
    text : Text to display as header

    Returns
    -------
    A header defined by html5 code below.

    '''
    html_temp = f"""
    <h2 style = "color:#F26531; text_align:center; font-weight: bold;"> {text} </h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)

def sub_text(text):
    '''
    A function to neatly display text in app.

    Parameters
    ----------
    text : Just plain text.

    Returns
    -------
    Text defined by html5 code below.

    '''
    
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)
    

def expander():
    '''
    
    Use Streamlit expander API and neatly show references.
    Call sub_text function.

    Returns
    -------
    An expander with special mentions and references.

    '''
    
    expander_text = """
        <p> Object detection functionality &#10024 </p>
        <p> The model used is: <b>ssd_mobiledet_cpu_coco_int8.tflite</b>, downloaded from 
        <a href= 'https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/MobileDet_Conversion_TFLite.ipynb#scrollTo=_rz1wbDv58t2' style="color:#F26531;" >this google colab</a>.
        The author of this notebook is
        <a href = 'https://github.com/sayakpaul' style="color:#F26531;" > sayakpaul </a>. </p>
        <p> Snapshot functionality &#128247 </p>
        <p> The webrtc snapshot functionality was shared in 
        <a href = 'https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669/23?u=whitphx'  style="color:#F26531;" > this discussion </a> 
        by the author of the component 
        <a href = 'https://github.com/whitphx'  style="color:#F26531;" > whitphx </a>. </p> 
        <p> Thanks so much to both authors and their amazing work &#129330 </p> 
        """
    
    expander = st.beta_expander("References & special thanks", expanded=False)
    
    with expander:
        
        sub_text(expander_text)

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
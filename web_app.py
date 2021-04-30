# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:41:36 2021

@author: TNIKOLIC
"""

import base64
import streamlit as st
import queue

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
#import cv2 as cv
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

#FRAME_WINDOW = st.image([])

def set_bg_hack():
    # set bg name
    main_bg = "ys_background.png"
    main_bg_ext = "png"
    
    # we can add a side bg if necessary 
    #side_bg = "sample.jpg"
    #side_bg_ext = "jpg"
        
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
    <h2 style = "color:#228B22; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#008B8B; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

def app_sendonly():
    """A sample to use WebRTC in sendonly mode to transfer frames
    from the browser to the server and to render frames via `st.image`."""
    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS
    )
    
    if webrtc_ctx.video_receiver:
        image_loc = st.empty()
        while True:
            try:
                frame = webrtc_ctx.video_receiver.get_frame(timeout=2)
                img_rgb = frame.to_ndarray(format="rgb24")
                image_loc.image(img_rgb)
                
            except queue.Empty:
                print("Queue is empty. Stop the loop.")
                webrtc_ctx.video_receiver.stop()
                return img_rgb
                break

            img_rgb = frame.to_ndarray(format="rgb24")
            image_loc.image(img_rgb)
        
# main app setup 
try:
    
   set_bg_hack()
    
   # Main panel setup
   display_app_header(main_txt='YS Community CleanUp',
                      sub_txt='Clean up your community!')
    
   # add input UN ---
   
   img_rgb = app_sendonly()
   
   if img_rgb is None:
       st.write("Take a photo!")
   else: 
       st.image(img_rgb)
   
   # add leaderboard ---
  
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
#except TypeError:
     #st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:41:36 2021

@author: TNIKOLIC
"""

import base64
import threading
from typing import Union
from pil import Image
import av
import numpy as np
import streamlit as st
#import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from datetime import date
import time


def object_detection(image):
    
    from garbage_detection import GarbageImageClassifier
    from IPython.display import Image
    
    GarbageImageClassifier = GarbageImageClassifier(cuda=False)
    
    attributes = GarbageImageClassifier.detect_image(image)
    
    if len(attributes) == 0: 
        
        st.write("No trashbags were detected in this photo.")
        
    else:

        st.write(attributes)
        
        st.write(attributes[0]['counts'])
    
    #python detector_garb.py -i samples/input5_frame11.jpg -o output

def main():
    
    # Main panel setup
    display_app_header(main_txt='YS Community CleanUp Project',
                      sub_txt='Welcome to the YSCC App. This app allows you to take photos of your surroundings while cleaning up.')
    
    set_bg_hack()
    
    image = Image.open('logo.png')
    st.image(image)
    
    username = st.text_input("Please input username", 
                             "example_1")
    
    if username != 'example_1': 
        
        
        gdpr_text = """
        <br> The aim of this <b>Data Protection Notice</b> is to provide you with 
        all the relevant information regarding the collection and further processing 
        of your Personal Data by Sogeti/Capgemini in the context of this project.</br>
        <br><b>Key data protection notions</b></br>
        <br>“Personal data” does not only refer to information related to your private life 
        but encompasses any and all information which enables to identify you either directly 
        or indirectly even where collected in a business and/or employment context.</br>
        <br>“Processing” means any operation which is performed on personal data, such as collection,
         recording, organization, structuring, storage, adaptation or alteration, retrieval, 
        consultation, use, disclosure, combination, restriction, erasure, or destruction.</br>
        <br>“Controller” means the natural or legal person which determines the purposes and means of 
        the processing of personal data. </br>
        <br>“Processor” means the natural or legal personal which processes personal data on behalf 
        of the controller. </br>
        <br>“Purpose” means the reason(s) why the controller needs to collect and further process 
        the personal data.</br>
        <br><b>Who is collecting your Personal Data?</b></br>
        <br>Sogeti Netherlands BV as part of the Capgemini Group hereafter referred to as “Sogeti”, 
        are collecting and further processing your personal data in their respective capacity as 
        data controllers.</br>
        <br><b>Which personal data is being processed?</b></br>
        <br>Initial user input, i.e. email address and username. The username will be displayed on 
        the leader board. The photos uploaded via the app, along with metadata information 
        attached to them shall also be processed with key data protection notions in place.</br>
        <br><b>Why and on what ground is Sogeti collecting your personal data?</br></b>
        <br>Sogeti is collecting and further processing your personal data for several reasons, 
        each of which is based on a specific legal ground as defined hereunder:</br>
        <br>- Main Purpose(s) </br>
        <br>- Legal ground(s) </br>
        <br>- Employee Engagement </br>
        <br>- Legitimate Interest of Sogeti as a Controller / Employer </br>
        <br>- Creating a sense of community </br>
        <br>- Environmental awareness Consent. </br>
        <br><b>Who has access to your personal data?</b></br>
        <br>Sogeti shall have access to your personal data. However, such access shall be strictly 
        limited to the relevant stakeholder(s) both from a functional and geographical scope. 
        As a result, for the above-mentioned purposes, your personal data will be shared 
        mainly with the following functions only on a need to know basis: Staffing and their 
        competent managers.</br>
        <br><b>How long does Sogeti keep your personal data?</b></br>
        <br>The data will be kept for 4 weeks, starting from the end of event (4th of June to 4th of July). 
        Sogeti shall keep your personal data for no longer than the duration of your employment with 
        Sogeti Nederland.</br>
        <br><b>What are your rights and how to exercise them?</b></br>
        <br>You can request to access, rectify, or erase your personal data. You may also object to the 
        processing of your personal data, or request that it be restricted. In addition, you can 
        ask for the communication of your personal data in a structured, commonly used and machine-readable 
        format.</br>
        <br>If you wish to exercise those rights, please contact our Global Data Protection Office by sending 
        an email to the following address: dpocapgemini.global@capgemini.com.
        Please note that you also have the right to lodge a complaint before a data protection authority or 
        the competent court of law.</br>
        """
        
        display_app_header("GDPR Prompt", gdpr_text, is_sidebar=True)
        
        # add sidebar gdpr prompt
        gdpr_prompt = st.sidebar.selectbox(
            "Select one of the following:",
            ("None", "Accept", "Decline")
        )
        
        if gdpr_prompt == 'None':
            
            display_app_header("Please respond to the GDPR prompt.",
                               "")
            
        elif gdpr_prompt == 'Decline':
            
            display_app_header("Sorry to see you leave!",
                               "We are here if you chage your mind.")
            
        else:
    
            option = st.selectbox(
                'Type of photo',
                ('None', 'Surroundings', 'Trashbags'))
            
            if option != 'None':
           
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
                            
                            file_name = "./output/" + username + "_" + option + "_photo_" + d + "_" + current_time + ".jpg"
                            
                            out_image = Image.fromarray(out_image)
                            
                            out_image.save(file_name)
                            
                            display_app_header("Object detection:",
                                               "")
                            object_detection(file_name)
                        else:
                            display_app_header("No frames available yet.",
                                               "")
                            
            else:
                display_app_header("Please select the type of photo you would like to take.",
                                   "")


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


if __name__ == "__main__":
    main()
        

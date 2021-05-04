# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:41:36 2021

@author: TNIKOLIC
"""

import base64
import threading
from typing import Union

import av
import numpy as np
import streamlit as st

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


def main():
    
    set_bg_hack()
    
    # Main panel setup
    display_app_header(main_txt='YS Community CleanUp',
                      sub_txt='Clean up your community!')
   
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
                st.write("Input image:")
                st.image(in_image, channels="BGR")
                st.write("Output image:")
                st.image(out_image, channels="BGR")
            else:
                st.warning("No frames available yet.")


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
        

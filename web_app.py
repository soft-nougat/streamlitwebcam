# -*- coding: utf-8 -*-
"""

YS Community CleanUp App 

Created on Mon Apr 19 18:45:07 2021

@author: TNIKOLIC
"""

import streamlit as st
import base64
from webcam import webcam
import SessionState

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.png')

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
    <h2 style = "color:#F74369; text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#BB1D3F; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else: 
        st.markdown(html_temp, unsafe_allow_html = True)

# app setup 
try:
    
    # set bg
    set_png_as_page_bg("app_bg.png")
    
    # Main panel setup
    display_app_header(main_txt='YS Community CleanUp',
                       sub_txt='Clean up your community!')
    
    # hide warning for st.pyplot() deprecation
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    captured_image = webcam()
    if captured_image is None:
        st.write("Waiting for capture...")
    else:
        st.write("Got an image from the webcam:")
        st.image(captured_image)
    
    
    
except ValueError:
    st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
    
except TypeError:
     st.error("Oops, something went wrong. Please check previous steps for inconsistent input.")
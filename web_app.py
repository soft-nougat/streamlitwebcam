# -*- coding: utf-8 -*-
"""

YS Community CleanUp App 

Created on Mon Apr 19 18:45:07 2021

@author: TNIKOLIC
"""
import pandas as pd
import streamlit as st
import base64
from webcam import webcam
import SessionState

# ----------------------------------------------
# session state
# needs to be refined, session state is used to
# successfully cache objects so the app runs
# smoothly
ss = SessionState.get(output_df = pd.DataFrame(), 
    df_raw = pd.DataFrame(),
    _model=None,
    text_col='text',
    is_file_uploaded=False,
    id2word = None, 
    corpus= None,
    is_valid_text_feat = False,
    to_clean_data = False,
    to_encode = False,
    to_train = False,
    to_evaluate = False,
    to_visualize = False,
    to_download_report = False,
    df = pd.DataFrame(),
    txt = 'Paste the text to analyze here',
    default_txt = 'Paste the text to analyze here',
    clean_text = None,
    ldamodel = None,
    topics_df = None)


# set background, use base64 to read local file
def get_base64_of_bin_file(bin_file):
    """
    function to read png file 
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """
    function to display png as bg
    ----------
    png_file: png -> the background image in local folder
    """
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
    #set_png_as_page_bg("app_bg.png")
    set_png_as_page_bg('background.png')

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
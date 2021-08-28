# Streamlitwebcam

This repository is home to the webrtc snapshot object detection Stremlit app. That was a mouthful, right? :D

The tutorial for building this app is published on [Medium](https://medium.com/sogetiblogsnl/streamlit-to-the-rescue-7d5f2f663465).

Deployed app on streamlit sharing is [here](https://share.streamlit.io/soft-nougat/streamlitwebcam/main/web_app.py). 

# App parts

The app contains 2 major blocks - the webrtc snapshot component and the tf lite object detection part.

## Object detection functionality ✨

The model used is: ssd_mobiledet_cpu_coco_int8.tflite, downloaded from [this](https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/MobileDet_Conversion_TFLite.ipynb#scrollTo=_rz1wbDv58t2) google colab. The author of this notebook is sayakpaul.

## Snapshot functionality 📷

The webrtc snapshot functionality was shared in [this](https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669/23?u=whitphx) discussion by the author of the component whitphx.

Thanks so much to both authors and their amazing work 🤲

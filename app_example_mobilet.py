import streamlit as st
import cv2 
import numpy as np
import time
import os
from utils.myDetector import myCustomDetector

#vanessa.ferrando@gmail.com
#dkleita@gmail.com
#enen96.maroc@gmail.com

# Windows 
st.title("Setup Configuration")
st.subheader(" Setup")
st.text("Ciao Manuel")
code = '''
    virtualenv env

    Windows: .\env\Scripts\activate
    Linux/Mac: source env/bin/activate

    pip streamlit==0.62 plotly==4.12
    pip install opencv-python
    '''

st.code(code, language='bash')

choice = st.radio(label="", options=["Start","Stop"], index=1)
running=False
if choice=="Start":
    st.subheader("Mask Detector: Started")

    cap = cv2.VideoCapture(0)
    
    mydetector = myCustomDetector()
    
    image_placeholder = st.empty()
    text_placeholder = st.empty()
    running= True
    while(running):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res, frame, percentage = mydetector.doDetection(frame)

            if res:
                text_placeholder.header("Persona trovata: %s" % (str(percentage)))
            else:
                text_placeholder.header("Nessuna persona")
                
            image_placeholder.image(frame)
            time.sleep(0.033)
    cap.release()

else:
    # non necessario streamlit riesegue il codice dall'inizio
    running = False
    st.subheader("Mask Detector: Stopped")




# API REFERENCE: https://docs.streamlit.io/en/stable/api.html
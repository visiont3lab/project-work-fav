import streamlit as st
import cv2 
import time
import numpy as np
import os
from pydub import AudioSegment
from pydub.playback import play

def findLargestBB(bbs):
    areas = [w*h for x,y,w,h in bbs]
    if not areas:
        return False, None
    else:
        i_biggest = np.argmax(areas) 
        biggest = bbs[i_biggest]
        return True, biggest

def page_mask():
    
    st.title("Mask Detector")

    cap = cv2.VideoCapture(0)

    choice = st.radio(label="", options=["Start","Stop"], index=1)
    if choice =="Stop":
        st.subheader("Mask Detector: Stopped")
        cap.release()
    
    else:
        
        st.subheader("Mask Detector: Started")

        cascade_masks = cv2.CascadeClassifier(os.path.join('models','mask_cascade.xml'))
        cascade_faces = cv2.CascadeClassifier(os.path.join('models','haarcascade_frontalface_default.xml'))
        
        text_placeholder = st.empty()
        image_placeholder = st.empty()

        speak_no = 0
        speak_yes = 0
        
        person_detected = 0

        while(True):
            # Capture frame-by-frame
            ret, im_color = cap.read()


            # Our operations on the frame come here
            im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)


            masks = cascade_masks.detectMultiScale(im_gray, 1.1,4,cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
            faces = cascade_faces.detectMultiScale(im_gray, 1.1,4,cv2.CASCADE_DO_ROUGH_SEARCH | cv2.CASCADE_SCALE_IMAGE)
            
            resMasks, biggestMask = findLargestBB(masks)
            resFaces, biggestFace = findLargestBB(faces)
            if resFaces:
                (x,y,w,h) = biggestFace
                roi = im_gray[y:y+h,x:x+w]
                cv2.rectangle(im_color,(x,y),(x+w,y+h),(255,255,0),2)

            if resMasks or resFaces:
                person_detected = person_detected + 1
            else:
                person_detected = 0
        
            if person_detected >=3:
                if resMasks:
                    (x,y,w,h) = biggestMask
                    roi = im_gray[y:y+h,x:x+w]
                    cv2.rectangle(im_color,(x,y),(x+w,y+h),(255,0,0),2)
        
                    text_placeholder.success("Mascherina indossata correttamente")
                    speak_yes = speak_yes + 1
                    if speak_yes > 10:
                        song = AudioSegment.from_wav(os.path.join("audio","procedere.wav"))
                        play(song)
                        speak_yes=0

                    speak_no=0
                else:
                    text_placeholder.error("Indossare la Mascherina")
                    speak_no = speak_no + 1
                    if speak_no > 10:
                        song = AudioSegment.from_wav(os.path.join("audio","indossare_mascherina.wav"))
                        play(song)
                        speak_no=0

                    speak_yes = 0

            image_placeholder.image(im_color, channels="BGR", use_column_width=True)
            time.sleep(0.033) # 30 Hz


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier(os.path.join('models','mask_cascade.xml'))

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, im_color = cap.read()


        # Our operations on the frame come here
        im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(im_gray, 1.1,5)

        for (x,y,w,h) in faces:
            roi = im_gray[y:y+h,x:x+w]
            cv2.rectangle(im_color,(x,y),(x+w,y+h),(255,0,0),2)


        # Display the resulting frame
        cv2.imshow('frame',im_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


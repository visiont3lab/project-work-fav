import streamlit as st
from utils.utils import get_log_image, get_current_date
from page_mask import page_mask
from page_covid import page_covid



def main():

	INTRODUCTION = "Introduction"
	MASKAPP = "MaskApp"
	COVIDAPP = "CovidApp"
	options_pages = [INTRODUCTION, MASKAPP, COVIDAPP]
	choice_pages= st.sidebar.radio(label="Pages",options=options_pages, index=0) # Index=0 = Login default choice	

	if choice_pages==INTRODUCTION:
		# set title
		st.title("FAV: Project Work %s" % (get_current_date()) )
		# get image logp
		#res, image = get_log_image()
		#if res: st.image(image, caption='',use_column_width=True)

		st.markdown(
			'''
			
			## Setup
			```
			sudo apt install python3-pip python3-dev virtualenv 
			virtualenv --python=python3 env
			source env/bin/activate
			#pip install opencv-python streamlit pydub pyaudio plotly==4.12
			pip install -r requirements.txt
			# Note: you can use opencv-pỳthon-headless version for server (does not have gui)
			```
			
			## Run
			
			```
			On Your PC or Colab: streamlit run app.py
			Procfile Heroku:  web: streamlit run --server.enableCORS false --server.port $PORT app.py
			'''


		)

	elif choice_pages==MASKAPP:
		page_mask()
	else:
		page_covid()
	
if __name__ == '__main__':
	main()
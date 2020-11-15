# Streamlit Project  Work App
		
## PC Setup

```
sudo apt install python3-pip python3-dev virtualenv 
virtualenv --python=python3 env
source env/bin/activate
#pip install opencv-python streamlit==0.62 pydub pyaudio plotly==4.12
pip install -r requirements.txt
```
* Note: you can use opencv-pỳthon-headless version for server (does not have gui)
* Note: we use streamlit==0.62 for compatibility with rasberry (0.71 streamlit version not working (pyarrow error))
* https://discuss.streamlit.io/t/raspberry-pi-streamlit/2900

## Run

```
On Your PC or Colab: streamlit run app.py
Procfile Heroku:  web: streamlit run --server.enableCORS false --server.port $PORT app.py

## Rasberry PI installation steps

```
sudo apt install python3-dev python3-pip 
sudo apt-get install libatlas-base-dev
sudo apt install python3-opencv
sudo pip3 install -U streamlit==0.62 plotly==4.12
sudo pip3 install -U pyhub pyaudio
```

* Known Issue streamlit==0.71:  [Failed building wheel for pyarrow ](https://discuss.streamlit.io/t/raspberry-pi-streamlit/2900/19) This is why we use streamlit==0.62

## Interesting link
* [Mqqt Streamlit Streaming](https://github.com/robmarkcole/mqtt-camera-streamer)
* [Udacity Self Driving Car](https://github.com/streamlit/demo-self-driving)

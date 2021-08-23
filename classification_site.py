from librosa.core import audio
from matplotlib.pyplot import text
import streamlit as st
import utils
import copy
import numpy as np
import pandas as pd

"""
Motor-sounds classification
"""

if "index" not in st.session_state:
  st.session_state.index = 0


sound_files = st.sidebar.file_uploader("Upload .wav file of motor sound", type="wav", accept_multiple_files=True, help="Sound file should be recording of motor and its length should be 6 seconds.")
label_file = st.sidebar.file_uploader("Upload .csv file of according class labels", type="csv")
copied_files = copy.deepcopy(sound_files)

if len(sound_files) == 1:
  if label_file is not None:
    label_file = pd.read_csv(label_file, index_col=0)
    st.session_state.index = label_file.loc[sound_files[0].name, 'ClassId'] + 1
  label_options = np.array([None, 0, 1, 2, 3, 4])
  select_label = st.sidebar.selectbox("Manually select class label", options=label_options, index=int(st.session_state.index), format_func=lambda x: utils.label_format.get(x), help="Select class label if not already determined by label file.\nIf label is not known, select None.")

if len(sound_files) >= 1:
  wave_fig = utils.plot_audio(sound_files)
  spectogram_fig = utils.plot_spectogram(sound_files)
  predicted, outputs = utils.predict_unknown_sample(copied_files[0])

display_style = st.sidebar.selectbox("Display style",options=("Multiple at once","Only one"))
if display_style == "Multiple at once":
  if len(sound_files) >= 1:
    audio_box = st.sidebar.checkbox("Audio")
    wave_box = st.sidebar.checkbox("Wave signal")
    spectogram_box = st.sidebar.checkbox("Spectogram")
  if len(sound_files) == 1:
    predict_box = st.sidebar.checkbox("Prediction")
  if len(sound_files)>= 1:
    if audio_box:
      for sound_file in sound_files:
        col1, col2 = st.columns([1,4])
        col1.write(sound_file.name)
        col2.audio(sound_file)
    if wave_box:
      st.pyplot(wave_fig)
    if spectogram_box:
      st.pyplot(spectogram_fig)
    if len(sound_files) == 1:
      if predict_box:
        st.pyplot(utils.plot_predictions(outputs, copied_files[0]))
        st.text(f"The predicted class of this motor is {int(predicted)}.")
        if select_label is not None:
          st.text(f"The actual class of this motor is {utils.label_format.get(select_label)}.")
          if int(predicted)==select_label:
            st.text("The prediction of the model is correct.")
          else:
            st.text("The prediction of the model is false.")
elif display_style == "Only one":
  if len(sound_files)> 1:
    display_select = st.sidebar.radio("",options=("Audio", "Wave signal", "Spectogram"))
  elif len(sound_files) == 1:
    display_select = st.sidebar.radio("",options=("Audio", "Wave signal", "Spectogram", "Prediction"))
  if len(sound_files)>= 1:
    if display_select == "Audio":
      for sound_file in sound_files:
        col1, col2 = st.columns([1,4])
        col1.write(sound_file.name)
        col2.audio(sound_file)
    elif display_select == "Wave signal":
      st.pyplot(wave_fig)
    elif display_select == "Spectogram":
      st.pyplot(spectogram_fig)
    if len(sound_files) == 1:
      if display_select == "Prediction":
        st.pyplot(utils.plot_predictions(outputs, copied_files[0]))
        st.text(f"The predicted class of this motor is {int(predicted)}.")
        if select_label is not None:
          st.text(f"The actual class of this motor is {utils.label_format.get(select_label)}.")
          if int(predicted)==select_label:
            st.text("The prediction of the model is correct.")
          else:
            st.text("The prediction of the model is false.")





  
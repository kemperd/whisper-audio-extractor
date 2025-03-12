import streamlit as st
import pandas as pd
import ffmpeg
import glob
import os
import whisper
from whisper.utils import get_writer

st.set_page_config(layout='wide')
st.title('Text extractor')

TEMP_OUTPUT_FILE = 'temp.mp3'
TEMP_VIDEO_FILE = 'temp_video'
AUDIO_DIR = 'audio'
SAMPLES_DIR = 'samples'

def load_tsv(filename):
    df = pd.read_csv(filename, sep='\t')
    return df

def extract_audio(df, line_num, input_filename, output_filename):
    start_ts = str(df.iloc[line_num].start - begin_padding) + 'ms'
    end_ts = str(df.iloc[line_num].end + end_padding) + 'ms'
    
    ffmpeg.input(input_filename, ss=start_ts, to=end_ts).output(output_filename).run(overwrite_output=True)

def select_row():
    pass  # To prevent Streamlit resetting to first tab

def convert_vid_to_mp3(mp3_filename):
    input = ffmpeg.input(TEMP_VIDEO_FILE)
    audio = input.audio
    out = ffmpeg.output(audio, AUDIO_DIR + '/' + mp3_filename)
    out.run(overwrite_output=True)

# Extracts the root filename without an extension
def extract_filename_root(filename):
    return os.path.splitext(filename)[0]

def extract_texts_from_mp3(mp3_filename, model_name):
    model = whisper.load_model(model_name)
    result = model.transcribe(mp3_filename)
    tsv_writer = get_writer('tsv', AUDIO_DIR)
    tsv_writer(result, mp3_filename)

# Configure sidebar
with st.sidebar:
   begin_padding = st.slider('Begin padding', min_value=0, max_value=3500, step=100, value=100)
   end_padding = st.slider('End padding', min_value=0, max_value=3500, step=100, value=500)

tab_conv_video, tab_extract_texts, tab_process_tsv = st.tabs(['Convert video', 'Extract texts', 'Process TSVs'])

## Video conversion tab
with tab_conv_video:
    uploaded_file = st.file_uploader('Select video')
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(TEMP_VIDEO_FILE, 'wb') as f:
            f.write(bytes_data)

        if st.button('Convert to MP3'):
            with st.spinner('Converting video...'):
                convert_vid_to_mp3(extract_filename_root(uploaded_file.name) + '.mp3')
                st.success('Done!')

## Text extraction tab
with tab_extract_texts:
    filelist = sorted(glob.glob(AUDIO_DIR + '/*.mp3'))
    mp3_filename = st.selectbox('Audio files', filelist, placeholder='Choose a file')

    if mp3_filename:
        model_name = st.selectbox('Model', ('turbo', 'large'))

        if st.button('Extract texts'):
            with st.spinner('Extracting texts...'):
                extract_texts_from_mp3(mp3_filename, model_name)
                st.success('Done!')

## TSV Processing tab
with tab_process_tsv:
    filelist = sorted(glob.glob(AUDIO_DIR + '/*.tsv'))
    if len(filelist) > 0:

        tsv_filename = st.selectbox('TSV files', filelist, placeholder='Choose a file')
        video_filename = extract_filename_root(tsv_filename) +  '.mp3'

        if tsv_filename:
            data = load_tsv(tsv_filename)

            selection = st.dataframe(
                data, 
                width=1200, 
                on_select=select_row, 
                selection_mode='single-row',
            )

            if st.button('Extract fragment'):
                id = selection['selection']['rows'][0]
                st.write('Extracting ' + str(id))
                extract_audio(data, id, video_filename, TEMP_OUTPUT_FILE)
                st.audio(TEMP_OUTPUT_FILE)

            if st.button('Save fragment'):
                id = selection['selection']['rows'][0]
                fragment_filename = SAMPLES_DIR + '/' + os.path.split(extract_filename_root(tsv_filename))[-1] + '_' + str(id) + '.mp3'
                st.write('Saving ' + fragment_filename)
                extract_audio(data, id, video_filename, fragment_filename)

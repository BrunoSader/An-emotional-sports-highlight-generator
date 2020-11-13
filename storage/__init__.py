from google.cloud import storage

from os import environ
from random import randint
import numpy as np
import sys

import cv2
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from python_speech_features import mfcc, logfbank, get_filterbanks
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


'''def storage_cfg(name, folder) :
    # Google Cloud Storage
    bucketName = environ.get(name)
    bucketFolder = environ.get(folder)
    tempFolder = environ.get('temp')
    return bucketName, bucketFolder, tempFolder

def list_files(bucket, bucketFolder):
    """List all files in GCP bucket."""
    files = bucket.list_blobs(prefix=bucketFolder)
    fileList = [file.name for file in files if '.' in file.name]
    return fileList'''

def view_video():
    cap = cv2.VideoCapture('storage/tmp/match.mkv')
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def video_to_audio(video_path, audio_path='storage/tmp/audio.wav'):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    return audio

if __name__ =='__main__' : 
    '''storage_client = storage.Client.from_service_account_json('storage/config/sport-highlight-pk.json')
    bucket = storage_client.get_bucket('football_matches')
    blob = bucket.get_blob('epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1.mkv')
    
    blob.download_to_filename('storage/tmp/match.mkv')'''

    #audio = video_to_audio('storage/tmp/match.mkv')
    ffmpeg_extract_subclip("storage/tmp/audio.wav", 0, 54000, targetname="storage/tmp/res_audio.wav")

    (rate,sig) = wav.read('storage/tmp/res_audio.wav')
    mfcc_feat = mfcc(sig,rate, nfft=1103)
    fbank_feat = logfbank(sig,rate, nfft=1103)
    print(len(fbank_feat))
    plt.plot(mfcc_feat)
    plt.show()
    plt.plot(fbank_feat)
    plt.show()


    # Deletes downloaded file from tmp
    # Comment the line if you wish to keep it
    os.remove('storage/tmp/match.mkv')




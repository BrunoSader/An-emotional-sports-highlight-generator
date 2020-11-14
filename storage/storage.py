from google.cloud import storage

import os
from random import randint
import numpy as np
import sys

import cv2
from moviepy.editor import *


def list_files(bucket, bucketFolder):
    """List all files in GCP bucket."""
    files = bucket.list_blobs(prefix=bucketFolder)
    fileList = [file.name for file in files if '.' in file.name]
    return fileList

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

def video_to_audio(video_path='storage/tmp/match.mkv', audio_path='storage/tmp/audio.wav', delete_video=False):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    if delete_video:
        os.remove('storage/tmp/match.mkv')
    return audio

def connect_db(file='storage/config/sport-highlight-pk.json') :
    return storage.Client.from_service_account_json('storage/config/sport-highlight-pk.json')

def get_video(storage_client, bucketName='football_matches', blobName='epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1.mkv', savePath='storage/tmp/match.mkv'):
    bucket = storage_client.get_bucket(bucketName)
    blob = bucket.get_blob(blobName)
    blob.download_to_filename(savePath)


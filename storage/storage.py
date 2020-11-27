from google.cloud import storage

import os
from random import randint
import numpy as np
import sys

import cv2
from moviepy.editor import *
from tqdm import tqdm


def list_files(storage_client, bucketName='football_matches', bucketFolder=''):
    bucket = storage_client.get_bucket(bucketName)
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

def video_to_audio_final(video_path='storage/tmp/highlights.mp4', audio_path='storage/tmp/audio_highlights.wav', delete_video=False):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)
    if delete_video:
        os.remove('storage/tmp/match.mkv')
    return audio

def connect_db(file='storage/config/sport-highlight-pk.json') :
    return storage.Client.from_service_account_json('storage/config/sport-highlight-pk.json')

def get_video(storage_client, blobName, bucketName='football_matches', savePath='storage/tmp/match.mkv'):
    bucket = storage_client.get_bucket(bucketName)
    blob = bucket.get_blob(blobName)
    blob.download_to_filename(savePath)

def upload_to_bucket(storage_client, blobName, bucketName='football_matches', uploadPath='storage/tmp/audio.wav'):
    bucket = storage_client.get_bucket(bucketName)
    blob = bucket.blob(blobName)
    blob.upload_from_filename(uploadPath)

def auto_upload_audio_to_bucket(storage_client, blobName, bucketName='football_matches', uploadPath='storage/tmp/audio.mkv'):
    for path in tqdm(list_files(storage_client)):
        print(path)
        get_video(storage_client, path)
        video_to_audio(delete_video=True)
        split_path = path.split('.')
        path = '{}.wav'.format(split_path[0])
        upload_to_bucket(storage_client,path)


def upload_folder_content():

    pathRoot='storage/tmp/AudioClasses/'
    folderList = ['Crowd', 'ExcitedCommentary', 'UnexcitedCommentary', 'Whistle']

    print("begin")

    for folder in tqdm(folderList):
        if(os.path.isdir(os.path.join(pathRoot, folder))):

            for filename in tqdm(os.listdir(os.path.join(pathRoot, folder))):
                filePath = pathRoot + folder + '/' + filename
                if(filename != '' and os.path.isfile(filePath)):
                    

    print("end")
                    

if __name__=='__main__' :
   
    '''
    client = connect_db()

    upload_folder_content()

    
    
    

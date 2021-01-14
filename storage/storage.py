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


def upload_folder_content(storage_client, blobName, bucketName, uploadPath='storage/tmp/AudioClasses/'):

    folderList = ['Crowd', 'ExcitedCommentary', 'UnexcitedCommentary', 'Whistle']

    for folder in tqdm(folderList):
        if(os.path.isdir(os.path.join(uploadPath, folder))):

            for filename in tqdm(os.listdir(os.path.join(uploadPath, folder))):
                filePath = uploadPath + folder + '/' + filename
                if(filename != '' and os.path.isfile(filePath)):

                    fileUploadPath = blobName + folder + '/' + filename
                    upload_to_bucket(storage_client, fileUploadPath, bucketName, filePath)
                    

def get_folder_content(storage_client, blobName, bucketName, savePath='storage/tmp/AudioClasses/'):

    # Get list of all files inside the bucketFolder
    bucket = storage_client.get_bucket(bucketName)
    files = bucket.list_blobs(prefix=blobName)
    fileList = [file.name for file in files if '.' in file.name]

    # Filter the prefix blobname prefix
    filteredList = []
    for file in fileList:

        if(blobName in file):
            newFileName = file.replace(blobName,'')
            filteredList.append(newFileName)

    #Create folders & download content to full path
    for index, file in enumerate(fileList):

        fullFilePath = savePath + filteredList[index]

        # Get final folder path
        indexLastSlash = filteredList[index].rfind("/")
        folderName = filteredList[index][0: indexLastSlash]
        finalSavePath = savePath + folderName
        
        # Create AudioClasses folder if doesn't exist
        if(not os.path.isdir(savePath)):
            os.mkdir(savePath)
        
        # Create classes folders if they don't exist
        if(not os.path.isdir(finalSavePath)):
            os.mkdir(finalSavePath)

        get_video(storage_client, file, bucketName, fullFilePath)


if __name__=='__main__' :

    client = connect_db()
    # get_video(client,'epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1.mkv')
    # video_to_audio()
    
    # filesList = list_files(client, bucketName='football_matches')
    # print(filesList)

    # upload_folder_content(client, blobName='classification/v5/TrainHmmNormal/', bucketName='football_matches', uploadPath='storage/tmp/AudioClasses/')

    # get_folder_content(client, blobName='classification/v3/AudioClasses/UnexcitedCommentary', bucketName='football_matches', savePath='storage/tmp/AudioClassesUnex/')

    # get_video(client, blobName='france_ligue-1/2016-2017/2017-02-10 - 22-45 Bordeaux 0 - 3 Paris SG/2.mkv', bucketName='football_matches', savePath='storage/tmp/matchBordeauxPSG2.mkv')

    # video_to_audio(video_path='storage/tmp/matchPSGMetz.mkv', audio_path='storage/tmp/audio.wav', delete_video=False)

    
    
    

from google.cloud import storage
from os import environ
from random import randint
import numpy as np
import cv2

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

if __name__ =='__main__' : 
    storage_client = storage.Client.from_service_account_json('storage/config/sport-highlight-pk.json')
    bucket = storage_client.get_bucket('football_matches')
    blob = bucket.get_blob('epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1.mkv')
    
    blob.download_to_filename('storage/tmp/match.mkv')
    cap = cv2.VideoCapture('storage/tmp/match.mkv')

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Deletes downloaded file from tmp
    # Comment the line if you wish to keep it
    os.remove('storage/tmp/match.mkv')




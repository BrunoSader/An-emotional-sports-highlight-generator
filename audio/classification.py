from python_speech_features import mfcc, logfbank
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import itertools
import os, shutil
import math
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.editor import AudioFileClip


class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=10, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type,n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type') 

    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))
        
    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)


# Parse the input folder, create HMM model and return it
def getHmmModel(input_folder):

    hmm_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):

        # Get the name of the subfolder
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []
        
        # Iterate through the audio files (leaving 1 file for testing in each class)
        # for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:1]:
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:

                # Read the input file
                filepath = os.path.join(subfolder, filename)
                sampling_freq, audio = wavfile.read(filepath)
                
                # Extract MFCC features
                mfcc_features = mfcc(audio, sampling_freq)
                
                # Append to the variable X
                if len(X) == 0:
                    X = mfcc_features
                else:
                    X = np.append(X, mfcc_features, axis=0)

                # Append the label
                y_words.append(label)
        
        print('X.shape =', X.shape)
        
        # Train and save HMM model
        hmm_trainer = HMMTrainer(n_components=10)
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
    
    return hmm_models


def splitAudio(start, indexRange, segmLength, audioSegmPath):

    # indexRange = 565
    for index in range(indexRange):
        ffmpeg_extract_subclip("storage/tmp/audio.wav", start+index*segmLength, start+(index+1)*segmLength, targetname= audioSegmPath + str(start+index*segmLength) + ".wav")


def splitAudioNew(start, indexRange, audioSegmPath):
    
    audio = AudioFileClip(audioSegmPath)
    audioClips = []
    for index in range(indexRange):
        audioClips.append(audio.subclip(start+index*5, start+(index+1)*5))

    return audioClips


# Get video segments + lists of segments of all classes
def getInitialClasses(segmLength, classifResults):
    
    crowd = []
    excitedCom = []
    unexcitedCom = []

    for results in classifResults:

        if(results[1] == 'Crowd'):
            crowd.append(results[0])
        
        if(results[1] == 'ExcitedCommentary'):
            excitedCom.append(results[0])
        
        if(results[1] == 'UnexcitedCommentary'):
            unexcitedCom.append(results[0])

    return crowd, excitedCom, unexcitedCom


# Adds extra segments between important moments; treats isolated crowd and excited commentary
def fineTuneSelection(crowd, excitedCom):

    # Concatenate lists of usefull classes : list of [segment index,  class]
    useClasses = []
    for segment in excitedCom:
        useClasses.append([segment, 'ExcitedCommentary'])
    
    for segment in crowd:
        useClasses.append([segment, 'Crowd'])
    
    useClassesSort = sorted(useClasses, key=lambda result: result[0])


    # Add segment in between 2 large useful segments
    for index in range(len(useClassesSort)):
        if( index < (len(useClassesSort)-1) and useClassesSort[index][0] + 2*segmLength == int(useClassesSort[index+1][0]) and
        (useClassesSort[index][1] == 'Crowd' or useClassesSort[index][1] == 'ExcitedCommentary') and
        (useClassesSort[index+1][1] == 'Crowd' or useClassesSort[index+1][1] == 'ExcitedCommentary')):
            useClassesSort.append([ str(useClassesSort[index][0] + segmLength), 'Extra'])
    
    
    # Isolated short (1 window) excited commentary -> take previous segment
    for index in range(len(useClassesSort)):
        if(index > 1 and index < (len(useClassesSort)-1) and useClassesSort[index][1] == 'ExcitedCommentary' and 
        int(useClassesSort[index-1][0]) < useClassesSort[index][0] - 2*segmLength and useClassesSort[index][0] + 2*segmLength < int(useClassesSort[index+1][0])):
            useClassesSort.append([ str(useClassesSort[index][0] - segmLength), 'Extra'])


    # Isolated short (1 window) crowd cheering -> delete segment
    finalList = []
    for index in range(len(useClassesSort)):
        if(index > 1 and index < (len(useClassesSort)-1) and useClassesSort[index][1] == 'Crowd' and 
        int(useClassesSort[index-1][0]) < useClassesSort[index][0] - 2*segmLength and useClassesSort[index][0] + 2*segmLength < int(useClassesSort[index+1][0])):
            continue
        else:
            finalList.append([ str(useClassesSort[index][0]), useClassesSort[index][1]])

    # Sort final list
    finalListSort = sorted(finalList, key=lambda result: int(result[0]))

    return finalListSort


# Get video segments + lists of segments of all classes
def splitVideo(segmLength, finalList, videoPath):
    
    video = VideoFileClip(videoPath)
    videoClips = []

    for result in finalList:
        if(result[1] == 'Crowd' or result[1] == 'ExcitedCommentary' or result[1] == 'Extra'):
            videoClips.append(video.subclip(int(result[0]), int(result[0])+segmLength))

    return videoClips


# Create folder if it doesn't exist
# Delete content of existing folder 
def prepareFolder(folderPath):

    if(os.path.isdir(folderPath)):

        # Deleting files inside the folder
        for filename in os.listdir(folderPath):
            filePath = os.path.join(folderPath, filename)
            try:
                if os.path.isfile(filePath) or os.path.islink(filePath):
                    os.unlink(filePath)
                elif os.path.isdir(filePath):
                    shutil.rmtree(filePath)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (filePath, e))
    
    else :
        os.mkdir(folderPath)


# Get class labels
def computeClassLabel(segments_folder, hmm_models):

    results = []

    # Read each clip & compute class label
    for fileName in os.listdir(segments_folder):
        sampling_freq, audio = wavfile.read(segments_folder+fileName)
        mfcc_features = mfcc(audio, sampling_freq)
        max_score = -9999999999999999999
        output_label = None
        
        scoresList = []

        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)

            if score > max_score:
                max_score = score
                output_label = label

            # scoresList.append(math.trunc(score))

        segmIndex = fileName[0:-4]
        results.append([int(segmIndex), output_label])
        # results.append([fileName, scoresList])
    
    # Sort results by segment index
    sortedResults = sorted(results, key=lambda result: result[0])

    return sortedResults


# Write class info to specified txt file
def writeToFile(file, className, segmList):

    file.write(className)
    for segment in segmList:
        file.write(str(segment) + " ")
    
    file.write("\n")


# Write final results in column
def writeInColumn(file, results):

    for result in results:
        file.write(result[1] + ": " + str(result[0]) + '\n')


if __name__ =='__main__' :
    
    input_folder = 'storage/tmp/AudioClasses/'
    
    # Create hmm model
    hmm_models = getHmmModel(input_folder)
    
    # Preparing the folder
    audioSegmPath = "storage/tmp/audioSegments/"
    videoPath = 'storage/tmp/match.mkv'
    prepareFolder(audioSegmPath)
    segmLength = 5

    # Split audio into segments
    splitAudio(start=0, indexRange=565, segmLength=segmLength, audioSegmPath=audioSegmPath)

    # Get class labels on audio segments
    audioResults = computeClassLabel(audioSegmPath, hmm_models)

    # Get the base classes from the audio
    crowd, excitedCom, unexcitedCom = getInitialClasses(segmLength=segmLength, classifResults=audioResults)

    # Add and delete segments where needed
    finalListSort = fineTuneSelection(crowd, excitedCom)

    clip = VideoFileClip("storage/tmp/match.mkv")
    videoClips = splitVideo(segmLength=segmLength, finalList=finalListSort, videoPath=videoPath)

    # Delete previous classes segments distribution
    if(os.path.isfile("storage/tmp/classesSegments.txt")):
        os.remove("storage/tmp/classesSegments.txt")
    
    # Write the classes distribution in line
    f= open("storage/tmp/classesSegments.txt","w+")
    writeToFile(f, "Crowd:", crowd)
    writeToFile(f, "ExcitedCommentary:", excitedCom)
    writeToFile(f, "UnexcitedCommentary:", unexcitedCom)
    f.close()

    # Delete previous final results file
    if(os.path.isfile("storage/tmp/finalResults.txt")):
        os.remove("storage/tmp/finalResults.txt")

    # Write final results in column
    f = open("storage/tmp/finalResults.txt","w+")
    writeInColumn(f, finalListSort)
    f.close()

    finalVideo = concatenate_videoclips(videoClips)

    # Delete previous highlight
    if(os.path.isfile("storage/tmp/highlights.mp4")):
        os.remove("storage/tmp/highlights.mp4")

    finalVideo.write_videofile("storage/tmp/highlights.mp4")

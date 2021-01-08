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
from scipy.io.wavfile import write
import pickle


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


# Returns the sample rate and list of audio windows with 1s interval between windows
def getAudioWindows(audio, winLength):

    sampleRate = audio.fps
    audioWindows = []
    
    for index in range(int(audio.duration - winLength)):
        audioWindows.append(audio.subclip(int(index), int(index+winLength)))
        # print(index, index+winLength)

    return sampleRate, audioWindows


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
def splitVideo(finalList, videoPath):
    
    video = VideoFileClip(videoPath)
    videoClips = []

    for result in finalList:
        if(result[2] == 'Crowd' or result[2] == 'ExcitedCommentary' or result[2] == 'Extra'):
            videoClips.append(video.subclip(int(result[0]), int(result[1])))

    return videoClips


# Create folder if it doesn't exist; Delete content of existing folder 
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


# Returns the classes of initial windows : index is the window's start
def getClassesInitWin(sampleRate, audioWindows, hmm_models):
    
    windowsClasses = []
    index = 0
    for audio in audioWindows:
        
        # # New read
        # mfcc_features = mfcc(audio, sampleRate)

        # # Convert audioFileClip to array 
        # audioArray = audio.to_soundarray()

        # Writing to wav and reading from it
        audio.write_audiofile("storage/tmp/segm.wav", sampleRate)
        sampling_freq, audioRead = wavfile.read("storage/tmp/segm.wav")

        # Remove the exported audio
        if(os.path.isfile("storage/tmp/segm.wav")):
            os.remove("storage/tmp/segm.wav")


        mfcc_features = mfcc(audioRead, sampling_freq)
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

        windowsClasses.append([int(index), output_label])
        # windowsClasses.append([index, scoresList])

        index += 1
    
    # Sort results by segment index
    sortedWinClasses = sorted(windowsClasses, key=lambda result: result[0])

    return sortedWinClasses


# Get class labels
def computeClassLabel(segments_folder, hmm_models):

    results = []

    # Read each clip & compute class label
    for fileName in os.listdir(segments_folder):
        sampling_freq, audio = wavfile.read(segments_folder+fileName)
        mfcc_features = mfcc(audio, sampling_freq)
        max_score = -9999999999999999999
        output_label = None
        
        # scoresList = []

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


# Returns distribution of class labels for each second
def getClassDistrib(winClasses, winLength):

    # List of [nbCrowd, nbExcited, nbUnexcited] with index = window's start
    distribBySec = []
    for index in range(len(winClasses)):
        distrib = [0, 0, 0]
        distribBySec.append(distrib)
    
    for delta in range(winLength - 1):
        distrib = [0, 0, 0]
        distribBySec.append(distrib)
    
    index = 0
    for window in winClasses:
        if(window[1] == 'Crowd'):
            for delta in range(winLength):
                distribBySec[index + delta][0] += 1
        
        if(window[1] == 'ExcitedCommentary'):
            for delta in range(winLength):
                distribBySec[index + delta][1] += 1

        if(window[1] == 'UnexcitedCommentary'):
            for delta in range(winLength):
                distribBySec[index + delta][2] += 1
        
        index += 1
    
    return distribBySec


# Puts a class label on each second based on max of distribution
def labelEachSec(distribBySec):

    classBySec = []
    labels = ['Crowd', 'ExcitedCommentary', 'UnexcitedCommentary']

    for distrib in distribBySec:
        maxIndex = distrib.index(max(distrib))
        classBySec.append(labels[maxIndex])
    
    return classBySec


# Concatenate identical class neigbor seconds into segments
def concatIntoSegments(classBySec):

    # List of [start, end, class]
    classSegments = []

    count = 0
    for index in range(len(classBySec)):
        if( index == 0 or classBySec[index] != classBySec[index-1] ):
            classSegments.append([index, index, classBySec[index]])
            if(count >= 1):
                classSegments[count-1][1] = index
            count += 1
    
    classSegments[count - 1][1] = len(classBySec)

    return classSegments


# Returns for each class type a list of corresponding segments
def getSegmByClass(classSegments):

    crowdSegm = []
    excitComSegm = []
    unexComSegm = []
    for segment in classSegments:
        if(segment[2] == 'Crowd'):
            crowdSegm.append([segment[0], segment[1]])
        
        if(segment[2] == 'ExcitedCommentary'):
            excitComSegm.append([segment[0], segment[1]])

        if(segment[2] == 'UnexcitedCommentary'):
            unexComSegm.append([segment[0], segment[1]])

    return crowdSegm, excitComSegm, unexComSegm


# Write class info to specified txt file
def writeToFile(file, className, segmList):

    file.write(className)
    for segment in segmList:
        file.write(str(segment) + " ")
    
    file.write("\n")


# Write final results in column
def writeInColumn(file, results):

    for index in range(len(results)):
        file.write(str(index) + ": " + str(results[index]) + '\n')


# Get the class that dominates in the scene
def classify_scene(video):

    segmLength = 5

    video.audio.write_audiofile(r"storage/tmp/audioScene.wav")

    # Read & cut audio
    audio = AudioFileClip("storage/tmp/audioScene.wav")

    # Delete audio file
    if(os.path.isfile("storage/tmp/audioScene.wav")):
        os.remove("storage/tmp/audioScene.wav")

    # Update segment length based on scene length
    if(audio.duration < segmLength):
        segmLength = audio.duration

    # Get hmm model
    with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)

    # Apply calculations
    sampleRate, audioWindows = getAudioWindows(audio, segmLength)
    windowsClasses = getClassesInitWin(sampleRate, audioWindows, hmm_models)
    distribBySec = getClassDistrib(windowsClasses, segmLength)
    classBySec = labelEachSec(distribBySec)

    # Get nb of occurence of each class in the scene
    classFreq = {"Crowd" : 0, "ExcitedCommentary" : 0, "UnexcitedCommentary" : 0}
    for second in classBySec:
        if(second == "Crowd"):
            classFreq["Crowd"] += 1
        
        if(second == "ExcitedCommentary"):
            classFreq["ExcitedCommentary"] += 1

        if(second == "UnexcitedCommentary"):
            classFreq["UnexcitedCommentary"] += 1
    
    # Get the label of the class with max occurence
    keyList = list(classFreq.keys())
    valList = list(classFreq.values())
    posOfMax = valList.index(max(valList))
    
    return keyList[posOfMax]
     


if __name__ =='__main__' :
    
    # Create hmm model
    hmm_models = getHmmModel('storage/tmp/AudioClasses/')

    # Write hmm model to pickle
    with open("audio/HmmModels.pkl", "wb") as file: pickle.dump(hmm_models, file)

    # video = VideoFileClip("storage/tmp/match.mkv")
    # dominantLabel = classify_scene(video)
    # print(dominantLabel)

    # # Get hmm model from pickle file
    # with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)
    
    # # Preparing the folder
    # audioSegmPath = "storage/tmp/audioSegments/"
    # videoPath = 'storage/tmp/match.mkv'
    # segmLength = 5

    # audio = AudioFileClip("storage/tmp/audio.wav")
    # sampleRate, audioWindows = getAudioWindows(audio, segmLength)
    # windowsClasses = getClassesInitWin(sampleRate, audioWindows, hmm_models)
    # distribBySec = getClassDistrib(windowsClasses, segmLength)
    # classBySec = labelEachSec(distribBySec)
    # classSegments = concatIntoSegments(classBySec)

    # # Delete previous initial classes file 
    # if(os.path.isfile("storage/tmp/initialClasses.txt")):
    #     os.remove("storage/tmp/initialClasses.txt")

    # # Write initial classes for windows
    # f= open("storage/tmp/initialClasses.txt","w+")
    # writeInColumn(f, windowsClasses)
    # f.close()


    # # Delete previous class distribution file
    # if(os.path.isfile("storage/tmp/classDistribution.txt")):
    #     os.remove("storage/tmp/classDistribution.txt")

    # # Write all segments distribution
    # f= open("storage/tmp/classDistribution.txt","w+")
    # writeInColumn(f, distribBySec)
    # f.close()


    # # Delete previous class by second
    # if(os.path.isfile("storage/tmp/classBySecond.txt")):
    #     os.remove("storage/tmp/classBySecond.txt")

    # # Write class by second distribution
    # f= open("storage/tmp/classBySecond.txt","w+")
    # writeInColumn(f, classBySec)
    # f.close()


    # # Delete previous class segments file (sequentially)
    # if(os.path.isfile("storage/tmp/sequentialClassSegments.txt")):
    #     os.remove("storage/tmp/sequentialClassSegments.txt")

    # # Write sequential class segments
    # f= open("storage/tmp/sequentialClassSegments.txt","w+")
    # writeInColumn(f, classSegments)
    # f.close()

    # videoClips = splitVideo(finalList=classSegments, videoPath=videoPath)
    # crowdSegm, excitComSegm, unexComSegm = getSegmByClass(classSegments)

    # # Delete previous classes segments distribution
    # if(os.path.isfile("storage/tmp/classesSegments.txt")):
    #     os.remove("storage/tmp/classesSegments.txt")
    
    # # Write the classes distribution in line
    # f= open("storage/tmp/classesSegments.txt","w+")
    # writeToFile(f, "Crowd:", crowdSegm)
    # writeToFile(f, "ExcitedCommentary:", excitComSegm)
    # writeToFile(f, "UnexcitedCommentary:", unexComSegm)
    # f.close()

    # finalVideo = concatenate_videoclips(videoClips)

    # # Delete previous highlight video
    # if(os.path.isfile("storage/tmp/highlights.mp4")):
    #     os.remove("storage/tmp/highlights.mp4")

    # finalVideo.write_videofile("storage/tmp/highlights.mp4")

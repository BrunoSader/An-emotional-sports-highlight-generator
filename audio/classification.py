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
from moviepy.editor import AudioFileClip, AudioClip
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
    
    if(audio.duration > winLength):
        for index in range(math.trunc(audio.duration)):
            audioWindows.append(audio.subclip(index, index+winLength))
    else:
        audioWindows.append(audio)

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
    windowsScores = []
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
            scoresList.append(math.trunc(score))

        windowsClasses.append([int(index), output_label])
        windowsScores.append([int(index), scoresList])

        index += 1
    
    # Sort results by segment index
    sortedWinClasses = sorted(windowsClasses, key=lambda result: result[0])
    sortedWinScores = sorted(windowsScores, key=lambda result: result[0])

    return sortedWinClasses, sortedWinScores


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

    winLength = math.trunc(winLength)

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

        if(distrib[1] >= distrib[0] and distrib[1] >= distrib[2]):
            classBySec.append(labels[1])
        elif(distrib[0] > distrib[1] and distrib[0] > distrib[2]):
            classBySec.append(labels[0])
        else:
            classBySec.append(labels[2])

        # maxIndex = distrib.index(max(distrib))
        # classBySec.append(labels[maxIndex])
    
    return classBySec


# Concatenate identical class neigbor seconds into segments
def concatIntoSegments(classBySec):

    # List of [start, end, class]
    classSegments = []

    count = 0
    for index in range(len(classBySec)):
        if( index == 0 or classBySec[index] != classBySec[index-1] ):
            classSegments.append([index, index, classBySec[index][1]])
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
def writeInColumn(file, results, startTime=0):

    for index in range(len(results)):
        file.write(str(startTime+index) + ": " + str(results[index]) + '\n')


# Get the class that dominates in the scene
def classify_scene(audio, segmLength, debug=False):

    if(segmLength > audio.duration):
        segmLength = math.trunc(audio.duration)
    
    if(segmLength == 0):
        segmLength = 1

    # Get hmm model
    with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)

    # Apply calculations
    sampleRate, audioWindows = getAudioWindows(audio, segmLength)
    windowsClasses = getClassesInitWin(sampleRate, audioWindows, hmm_models)
    distribBySec = getClassDistrib(windowsClasses, segmLength)
    classBySec = labelEachSec(distribBySec)

    if(debug == True):
        # Write initial classes for windows
        f= open("storage/tmp/initialClasses.txt","a")
        writeInColumn(f, windowsClasses)
        f.close()

        # Write all segments distribution
        f= open("storage/tmp/classDistribution.txt","a")
        writeInColumn(f, distribBySec)
        f.close()

        # Write class by second distribution
        f= open("storage/tmp/classBySecond.txt","a")
        writeInColumn(f, classBySec)
        f.close()

    # Get nb of occurence of each class in the scene
    classFreq = {"Crowd" : 0, "ExcitedCommentary" : 0, "UnexcitedCommentary" : 0}
    for second in classBySec:
        if(second == "Crowd"):
            classFreq["Crowd"] += 1
        elif(second == "ExcitedCommentary"):
            classFreq["ExcitedCommentary"] += 1
        elif(second == "UnexcitedCommentary"):
            classFreq["UnexcitedCommentary"] += 1
    
    # Get the label of the class with max occurence
    keyList = list(classFreq.keys())
    valList = list(classFreq.values())
    posOfMax = valList.index(max(valList))

    return keyList[posOfMax]


# Get the class that dominates in the scene
def classify_scene2(audio, startTime=0, debug=False):

    # Get hmm model
    with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)

    # Apply calculations
    audioWindows = []
    sampleRate = audio.fps
    if(audio.duration < 1):
        print("duration smaller than 1")
        audioWindows.append(audio)
    else:
        print("duration BIGGER than 1")
        sampleRate, audioWindows = getAudioWindows(audio, 1)
    classBySec, scoreBySec = getClassesInitWin(sampleRate, audioWindows, hmm_models)

    if(debug == True):
        # Write initial classes for windows
        f= open("storage/tmp/classBySecond.txt","a")
        writeInColumn(f, classBySec, startTime)
        f.close()

        # Write initial classes for windows
        f= open("storage/tmp/scoreBySecond.txt","a")
        writeInColumn(f, scoreBySec, startTime)
        f.close()

    # Get nb of occurence of each class in the scene
    classFreq = {"Crowd" : 0, "ExcitedCommentary" : 0, "UnexcitedCommentary" : 0, "Ambient": 0}
    for second in classBySec:
        if(second[1] == "Crowd"):
            classFreq["Crowd"] += 1
        elif(second[1] == "ExcitedCommentary"):
            classFreq["ExcitedCommentary"] += 1
        elif(second[1] == "UnexcitedCommentary"):
            classFreq["UnexcitedCommentary"] += 1
        elif(second[1] == "Ambient"):
            classFreq["Ambient"] += 1
    
    # Get the label of the class with max occurence
    keyList = list(classFreq.keys())
    valList = list(classFreq.values())
    posOfMax = valList.index(max(valList))

    print(classBySec)
    print("len=" + str(len(classBySec)))

    # if(keyList[posOfMax] == "ExcitedCommentary"):
    #     return "Save"
    # elif(len(classBySec) > 1 and classBySec[len(classBySec) - 1][1] == "ExcitedCommentary" and classBySec[len(classBySec) - 2][1] == "ExcitedCommentary"):
    #     return "Save"

    if(len(classBySec) == 1 and classBySec[len(classBySec) - 1][1] == "ExcitedCommentary"):
        return "SaveTheEnd", classBySec
    elif(len(classBySec) > 1 and classBySec[len(classBySec) - 1][1] == "ExcitedCommentary" and classBySec[len(classBySec) - 2][1] == "ExcitedCommentary"):
        return "SaveTheEnd", classBySec
    
    return "Pass", classBySec


#if __name__ =='__main__' :
    
    # Create hmm model
    # hmm_models = getHmmModel('storage/tmp/AudioClasses/')

    # # Write hmm model to pickle
    # with open("audio/HmmModels.pkl", "wb") as file: pickle.dump(hmm_models, file)

    # video = VideoFileClip("storage/tmp/match.mkv")
    # dominantLabel = classify_scene(video)
    # print(dominantLabel)

    # # Get hmm model from pickle file
    # with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)
    
    # # Preparing the folder
    # audioSegmPath = "storage/tmp/audioSegments/"
    # videoPath = 'storage/tmp/matchBordeauxPSG2.mkv'
    # segmLength = 5

    # audio = AudioFileClip("storage/tmp/audioBordeauxPSG2.wav")
    # sampleRate, audioWindows = getAudioWindows(audio, segmLength)
    # windowsClasses = getClassesInitWin(sampleRate, audioWindows, hmm_models)
    # distribBySec = getClassDistrib(windowsClasses, segmLength)
    # classBySec = labelEachSec(distribBySec)
    # classSegments = concatIntoSegments(classBySec)

    # # Write initial classes for windows
    # f= open("storage/tmp/initialClasses.txt","w+")
    # writeInColumn(f, windowsClasses)
    # f.close()

    # # Write all segments distribution
    # f= open("storage/tmp/classDistribution.txt","w+")
    # writeInColumn(f, distribBySec)
    # f.close()

    # # Write class by second distribution
    # f= open("storage/tmp/classBySecond.txt","w+")
    # writeInColumn(f, classBySec)
    # f.close()

    # # Write sequential class segments
    # f= open("storage/tmp/sequentialClassSegments.txt","w+")
    # writeInColumn(f, classSegments)
    # f.close()

    # videoClips = splitVideo(finalList=classSegments, videoPath=videoPath)
    # crowdSegm, excitComSegm, unexComSegm = getSegmByClass(classSegments)
    
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


    # ---------Test environment 1

    # audio = AudioFileClip("storage/tmp/largestScene.wav")
    # video = VideoFileClip("storage/tmp/matchBordeauxPSG2.mkv")

    # if(os.path.isfile("storage/tmp/initialClasses.txt")):
    #     os.remove("storage/tmp/initialClasses.txt")
    
    # if(os.path.isfile("storage/tmp/classDistribution.txt")):
    #     os.remove("storage/tmp/classDistribution.txt")

    # if(os.path.isfile("storage/tmp/classBySecond.txt")):
    #     os.remove("storage/tmp/classBySecond.txt")

    # if(os.path.isfile("storage/tmp/classByScene.txt")):
    #     os.remove("storage/tmp/classByScene.txt")

    # index = 0
    # winLen = 5
    # f= open("storage/tmp/classByScene.txt","a")
    # while (index < audio.duration):
    #     if(index + winLen < audio.duration):
    #         segment = audio.subclip(index, index + winLen)
    #         sceneResult = classify_scene(segment, winLen, True)
    #         f.write(str(index) + " " + str(index+winLen) + " " + sceneResult + "\n")
    #     index += winLen

    # f.close()
    
    # ---------Test on one scene
    
    # index = 0
    # winLen = 3
    # f= open("storage/tmp/classByScene.txt","a")
    
    # sceneResult = classify_scene(audio, winLen, True)
    # f.write(str(index) + " " + str(index+winLen) + " " + sceneResult + "\n") 

    # f.close()

    # ---------End test environment 1


    # ---------- Test environment 2
    # audio = AudioFileClip("storage/tmp/largestScene.wav")

    # if(os.path.isfile("storage/tmp/classBySecond.txt")):
    #     os.remove("storage/tmp/classBySecond.txt")

    # # if(os.path.isfile("storage/tmp/classByScene.txt")):
    # #     os.remove("storage/tmp/classByScene.txt")
    
    # sceneResult = classify_scene2(audio, True)
    # print(sceneResult + "\n")
    

    #  ---------- End test environment 2



    #  ---------- Test environment 3

    # # Create hmm model
    # hmm_models = getHmmModel('storage/tmp/AudioClasses1SecCuratedExcited/')

    # # Write hmm model to pickle
    # with open("audio/HmmModels.pkl", "wb") as file: pickle.dump(hmm_models, file)

    # # Get hmm model from pickle file
    # with open("audio/HmmModels.pkl", "rb") as file: hmm_models = pickle.load(file)

    # audio = AudioFileClip("storage/tmp/audioBordeauxPSG2.wav")
    # # video = VideoFileClip("storage/tmp/matchBordeauxPSG2.mkv")
    # sampleRate, audioWindows = getAudioWindows(audio, 1)
    # classBySec = getClassesInitWin(sampleRate, audioWindows, hmm_models)
    # classSegments = concatIntoSegments(classBySec)

    # # Write class by second distribution
    # f= open("storage/tmp/classBySecond.txt","w+")
    # writeInColumn(f, classBySec)
    # f.close()

    # # Write class by second distribution
    # f= open("storage/tmp/classSegments.txt","w+")
    # writeInColumn(f, classSegments)
    # f.close()

    # videoClips = splitVideo(finalList=classSegments, videoPath='storage/tmp/matchBordeauxPSG2.mkv')

    # finalVideo = concatenate_videoclips(videoClips)

    # # Delete previous highlight video
    # if(os.path.isfile("storage/tmp/highlights.mp4")):
    #     os.remove("storage/tmp/highlights.mp4")

    # finalVideo.write_videofile("storage/tmp/highlights.mp4")

    #  ---------- End test environment 3

    #  ---------- Test environment 4

    # Put together all scenes and sort them by the endScene

    # scenes = [ [2, 3], [7, 9], [13, 14], [20, 21], [28, 30]]
    # OCR_scenes = [ [1, 4], [8, 10], [15, 16], [24, 25] ]

    # allScenes = []
    # for scene in scenes:
    #     allScenes.append([scene[0], scene[1], "audio"])
    # for scene in OCR_scenes:
    #     allScenes.append([scene[0], scene[1], "ocr"])

    # sortedScenes = sorted(allScenes, key=lambda scene: scene[1])

    # # Algorithm for choosing the right scenes
    # near = 1
    # use = True
    # final_scenes = []
    # for i, scene in enumerate(sortedScenes):
    #     if(use == True):
    #         if(scene[2] == "audio" and i < len(sortedScenes) - 1 and sortedScenes[i+1][2] == "ocr"):
    #             if(sortedScenes[i+1][0] < scene[0]):
    #                 final_scenes.append(sortedScenes[i+1])
    #             elif( sortedScenes[i+1][0] > scene[0] and sortedScenes[i+1][0] <= scene[1] + near):
    #                 final_scenes.append(sortedScenes[i+1])
    #             else:
    #                 final_scenes.append(scene)

    #         elif(scene[2] == "audio" and i == len(sortedScenes) - 1 and (sortedScenes[i-1][2] == "audio" or sortedScenes[i-1][1] < scene[0] - near)):
    #             final_scenes.append(scene)
            
    #         elif(scene[2] == "audio"):
    #             final_scenes.append(scene)

    #         elif(scene[2] == "ocr" and i < len(sortedScenes) - 1):
    #             if(sortedScenes[i+1][0] < scene[0]):
    #                 final_scenes.append(scene)
    #                 use = False
    #             elif( sortedScenes[i+1][0] > scene[0] and sortedScenes[i+1][0] <= scene[1] + near):
    #                 final_scenes.append(sortedScenes[i+1])
    #             else:
    #                 final_scenes.append(scene)

    #         elif(scene[2] == "ocr" and i == len(sortedScenes) - 1 and (sortedScenes[i-1][2] == "ocr" or sortedScenes[i-1][1] < scene[0] - near)):
    #             final_scenes.append(scene)
            
    #         elif(scene[2] == "ocr"):
    #             final_scenes.append(scene)

    #     else:
    #         use = True

    # # Delete double cells
    # filteredScenes = []
    # for scene in final_scenes:
    #     if(len(filteredScenes) == 0 ):
    #         filteredScenes.append(scene)
    #     elif(len(filteredScenes) > 0 and filteredScenes[len(filteredScenes) - 1] != scene):
    #         filteredScenes.append(scene)
        
    # print(filteredScenes)

    #  ---------- End test environment 4
    
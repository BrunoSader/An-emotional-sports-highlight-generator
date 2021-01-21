import numpy as np
import argparse
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import time
import os
import glob
from collections import Counter, defaultdict

from video.scene_detection import detect_scene
from audio.classification import classify_scene, HMMTrainer, classify_scene2
# from audio.CNN_classifier import predict
from audio.spectro_analysis import concat_video
from ocr.final_ocr import ocr
from ocr.highlights_compare import generate_highlights_compare

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CNN", help="CNN, HMM")
parser.add_argument('--match-path', type=str, default="", help="path to match")
parser.add_argument('--fifa', action='store_true')
parser.add_argument('--demo', action='store_true')
args = parser.parse_args()

filename = args.match_path
capture = cv2.VideoCapture(filename)
fps = capture.get(cv2.CAP_PROP_FPS)
print('chosen model : {}'.format(args.model))
resizeWidth = 0
video = VideoFileClip(filename)
audio = video.audio
del video
fpf = int(audio.fps/fps) #frames per frame
i=-1
history = [0]
start_indices = [0]
last = None
frames = []
audioframes = []
scenes_count = 0
start = time.time()

winLen = 3
startScene = 0
endScene = 0

if(os.path.isfile("storage/tmp/initialClasses.txt")):
        os.remove("storage/tmp/initialClasses.txt")
    
if(os.path.isfile("storage/tmp/classDistribution.txt")):
    os.remove("storage/tmp/classDistribution.txt")

if(os.path.isfile("storage/tmp/classBySecond.txt")):
    os.remove("storage/tmp/classBySecond.txt")

if(os.path.isfile("storage/tmp/classByScene.txt")):
    os.remove("storage/tmp/classByScene.txt")

if(os.path.isfile("storage/tmp/deletedScenes.txt")):
    os.remove("storage/tmp/deletedScenes.txt")

if(os.path.isfile("storage/tmp/acceptedScenes.txt")):
    os.remove("storage/tmp/acceptedScenes.txt")

f= open("storage/tmp/classByScene.txt","a")
f1= open("storage/tmp/deletedScenes.txt","a")
f2= open("storage/tmp/acceptedScenes.txt","a")

if(os.path.isdir("storage/tmp/tmpMain/")):
    for filename in os.listdir("storage/tmp/tmpMain/"):
        filepath = os.path.join("storage/tmp/tmpMain/", filename)
        os.remove(filepath)

#TODO Add fifa mode
if args.model == 'CNN' :
    scene_description = 'Unknown' #used for demo
    for chunk in audio.iter_chunks(chunksize=fpf) : #simulates audio stream
        i+=1
        (grabbed, frame) = capture.read()
        if not grabbed:
            break

        if args.demo :
            cv2.imshow(scene_description,frame)

        # Resize frame to width, if specified
        if resizeWidth > 0:
            (height, width) = frame.shape[:2]
            resizeHeight = int(float(resizeWidth / width) * height)
            frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                interpolation=cv2.INTER_AREA)
        if i > 0 :
            if(detect_scene(frame, last)) :
                scene_classes = predict(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                d = defaultdict(list)
                for k, v in scene_classes:
                    d[v].append(k)
                l = [(i, np.mean(j)) for i,j in d.items()]
                score = dict(l)
                counter = Counter(elem[1] for elem in scene_classes[-3:]) #try for 2 - 5 seconds
                unexciting_condition = counter['Unexcited'] >= 2
                if args.fifa :
                    exciting_condition = counter['Excited'] > 0 or counter['Crowd'] > 0
                    crowd_condition = False
                else :
                    exciting_condition = counter['Excited'] > 0 and score['Excited'] > 0.80
                    crowd_condition = counter['Crowd'] > 0 and score['Crowd'] > 0.90
                if not unexciting_condition :
                    if exciting_condition :
                        if len(scene_classes) > 8 :
                            for i, item in enumerate(scene_classes[::-1]):
                                if item[1] == 'Unexcited':
                                    length = i+3
                                    frames = frames[int(-length*fps):]
                                    audioframes = audioframes[int(-length*audio.fps):]
                                    break
                        scene = ImageSequenceClip(frames, fps)
                        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                        scenes_count+=1
                        history.append(i)
                        scene_description= 'Excited'
                        frames.clear()
                        audioframes.clear()
                    elif not crowd_condition :   
                        scene_description= 'Unexcited' 
                        frames.clear()
                        audioframes.clear()
                    scene_description= 'Crowd'
                else :
                    scene_description= 'Unexcited'
                    frames.clear()
                    audioframes.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        audioframes.extend(chunk)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last = frame

    scene_classes = predict(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
    d = defaultdict(list)
    for k, v in scene_classes:
        d[v].append(k)
    l = [(i, np.mean(j)) for i,j in d.items()]
    score = dict(l)
    counter = Counter(elem[1] for elem in scene_classes[-3:]) #try for 2 - 5 seconds
    unexciting_condition = counter['Unexcited'] >= 2
    if args.fifa :
        exciting_condition = counter['Excited'] > 0 or counter['Crowd'] > 0
        crowd_condition = False
    else :
        exciting_condition = counter['Excited'] > 0 and score['Excited'] > 0.80
        crowd_condition = counter['Crowd'] > 0 and score['Crowd'] > 0.90
    if not unexciting_condition :
        if exciting_condition :
            if len(scene_classes) > 8 :
                for i, item in enumerate(scene_classes[::-1]):
                    if item[1] == 'Unexcited':
                        length = i+3
                        frames = frames[int(-length*fps):]
                        audioframes = audioframes[int(-length*audio.fps):]
                        break
            scene = ImageSequenceClip(frames, fps)
            scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
            scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
            scenes_count+=1
            history.append(i)
    frames.clear()
    audioframes.clear()

elif args.model == 'HMM' :

    sceneNameTimesRes = []

    for chunk in audio.iter_chunks(chunksize=fpf) : #simulates audio stream
        i+=1
        endScene = i
        (grabbed, frame) = capture.read()

        if not grabbed:
            break

        # Resize frame to width, if specified
        if resizeWidth > 0:
            (height, width) = frame.shape[:2]
            resizeHeight = int(float(resizeWidth / width) * height)
            frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                interpolation=cv2.INTER_AREA)
        if i > 0 :
            if(detect_scene(frame, last)) :
                
                # Write audio & read with audioclip
                clip = AudioArrayClip(np.asarray(audioframes), fps=audio.fps)
                clip.write_audiofile('storage/tmp/tmpMain/sceneTmp' + str(startScene/25) + '-' + str(endScene/25) + ' ' + '.wav', audio.fps)
                sceneAudio = AudioFileClip('storage/tmp/tmpMain/sceneTmp' + str(startScene/25) + '-' + str(endScene/25) + ' ' + '.wav')
                
                # Getting result action for the scene
                scene_class, classBySec = classify_scene2(sceneAudio, startScene/25, debug=True)
                sceneNameTimesRes.append(["", scene_class, startScene/25, endScene/25])

                f.write(str(startScene/25) + '-' + str(endScene/25) + ' ' + scene_class + '\n')
                
                # Append only interesting scenes
                if(scene_class == "SaveTheEnd"):
                    
                    savedFrames = []
                    savedAudioFrames = []

                    # Backward iteration to find nb of excited at the end
                    count = 0
                    for sec in reversed(classBySec):
                        if(sec[1] == "ExcitedCommentary"):
                            count += 1
                        else:
                            break
                    
                    if(len(classBySec) > count + 3):
                        count += 3
                    elif(len(classBySec) > count):
                        count += len(classBySec) - count
                    delta = len(classBySec) - count

                    savedFrames = frames[int(delta*fps) : len(frames)]
                    savedAudioFrames = audioframes[int(delta*audio.fps) : len(audioframes)]

                    scene = ImageSequenceClip(savedFrames, fps)
                    scene = scene.set_audio(AudioArrayClip(np.asarray(savedAudioFrames), fps=audio.fps))
                    scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                    sceneNameTimesRes[len(sceneNameTimesRes)-1][0] = "storage/tmp/scenes/scene{}.mp4".format(scenes_count)
                    scenes_count+=1
                    history.append(i)
                    start_indices.append(i-len(frames))

                f2.write(str(sceneNameTimesRes[len(sceneNameTimesRes)-1][0]) + ' ' + str(sceneNameTimesRes[len(sceneNameTimesRes)-1][1]) + ' ' + str(sceneNameTimesRes[len(sceneNameTimesRes)-1][2]) + ' ' + str(sceneNameTimesRes[len(sceneNameTimesRes)-1][3]) + '\n')

                frames.clear()
                audioframes.clear()
                startScene = i

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        audioframes.extend(chunk)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last = frame
        
    scene_class, classBySec = classify_scene2(AudioArrayClip(np.asarray(audioframes), fps=audio.fps), False)
    if(scene_class == "SaveTheEnd"):
        scene = ImageSequenceClip(frames, fps)
        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
        history.append(i)
        start_indices.append(i-len(frames))

    frames.clear()
    audioframes.clear()

    # Cut isolated-excited-too-short scenes
    for index in range(len(sceneNameTimesRes)):
        if( index > 1 and index < len(sceneNameTimesRes) - 1 and float(sceneNameTimesRes[index][3]) - float(sceneNameTimesRes[index][2]) <5 and 
        sceneNameTimesRes[index][1] == "SaveTheEnd" and sceneNameTimesRes[index-1][1] == "Pass" and sceneNameTimesRes[index+1][1] == "Pass"):
            
            f1.write( str(sceneNameTimesRes[index][0]) + " " + str(sceneNameTimesRes[index][1]) + " " +  str(sceneNameTimesRes[index][2]) + " " + str(sceneNameTimesRes[index][3]) + '\n')
            if(os.path.isfile(sceneNameTimesRes[index][0])):
                os.remove(sceneNameTimesRes[index][0])


concat_video(scenes_count, video_path='storage/tmp/scenes/scene', save_path='highlights.mp4')

end = time.time()

print("Elapsed time is  {}".format(end-start))

# files = glob.glob('storage/tmp/scenes/*')
# for f in files:
#     os.remove(f)

capture.release()
cv2.destroyAllWindows()

#ocr('ocr/img', filename, 'times.txt', 300)
#generate_highlights_compare('ocr/highlights_videos', 'secondmatch.mkv', 'ocr/img/times.txt', 10, start_indices)



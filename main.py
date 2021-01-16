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
from audio.classification import classify_scene, HMMTrainer
from audio.CNN_classifier import predict
from audio.spectro_analysis import concat_video
from ocr.final_ocr import ocr
from ocr.highlights import generate_highlights

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
last = None
frames = []
audioframes = []
scenes_count = 0
start = time.time()

# Delete previous class by second file
if(os.path.isfile("storage/tmp/classBySecond.txt")):
    os.remove("storage/tmp/classBySecond.txt")

#TODO Add fifa mode
if args.model == 'CNN' :
    scene_class = 'Unkown' #used for demo
    for chunk in audio.iter_chunks(chunksize=fpf) : #simulates audio stream
        i+=1
        (grabbed, frame) = capture.read()
        if not grabbed:
            break

        if args.demo :
            cv2.imshow(scene_class,frame)

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
                #unique_excited_condition = (len(scene_classes) <= 2 and 'Excited' in dict(counter.most_common(2))) 
                #unique_crowd_condition = (len(scene_classes) == 1 and 'Crowd' in dict(counter.most_common(1)) and score['Crowd'] > 0.90)
                #normal_condition = ('Excited' in dict(counter.most_common(1)) and score['Excited'] > 0.90)
                unexciting_condition = counter['Unexcited'] >= 2
                if args.fifa :
                    exciting_condition = counter['Excited'] > 0 or counter['Crowd'] > 0
                    crowd_condition = False
                else :
                    exciting_condition = counter['Excited'] > 0 and score['Excited'] > 0.90
                    crowd_condition = counter['Crowd'] > 0 and score['Crowd'] > 0.90
                if not unexciting_condition :
                    if exciting_condition :
                        if args.fifa :
                            frames = frames[int(-12*fps):]
                            audioframes = audioframes[int(-12*audio.fps):]
                        scene = ImageSequenceClip(frames, fps)
                        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                        scenes_count+=1
                        history.append(i)
                        scene_class = 'Excited'
                        frames.clear()
                        audioframes.clear()
                    elif not crowd_condition :   
                        scene_class = 'Unexcited' 
                        frames.clear()
                        audioframes.clear()
                    scene_class = 'Crowd'
                else :
                    scene_class = 'Unexcited'
                    frames.clear()
                    audioframes.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        audioframes.extend(chunk)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last = frame

    d = defaultdict(list)
    for k, v in scene_classes:
        d[v].append(k)
    l = [(i, np.mean(j)) for i,j in d.items()]
    score = dict(l)
    counter = Counter(elem[1] for elem in scene_classes)
    if('Excited' in dict(counter.most_common(1)) and score['Excited'] > 0.9):
        scene = ImageSequenceClip(frames, fps)
        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
        scenes_count+=1
        history.append(i)
    frames.clear()
    audioframes.clear()

elif args.model == 'HMM' :
    for chunk in audio.iter_chunks(chunksize=fpf) : #simulates audio stream
        i+=1
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
                scene_class = classify_scene(AudioArrayClip(np.asarray(audioframes), fps=audio.fps), debug=True)
                # Append only interesting scenes
                if(scene_class == "ExcitedCommentary"):
                    scene = ImageSequenceClip(frames, fps)
                    scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                    scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                    scenes_count+=1
                    history.append(i)
                frames.clear()
                audioframes.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        audioframes.extend(chunk)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last = frame

    scene_class = classify_scene(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
    if(scene_class == "Crowd" or scene_class == "ExcitedCommentary"):
        scene = ImageSequenceClip(frames, fps)
        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
        history.append(i)
    frames.clear()
    audioframes.clear()

concat_video(scenes_count, video_path='storage/tmp/scenes/scene', save_path='highlights.mp4')

end = time.time()

print("Elapsed time is  {}".format(end-start))

files = glob.glob('storage/tmp/scenes/*')
for f in files:
    os.remove(f)

capture.release()
cv2.destroyAllWindows()

#ocr('ocr/img', 'ocr/tmp/secondmatch.mkv', 'times.txt', 1080)
#generate_highlights('ocr/highlights_videos', 'secondmatch.mkv', 'ocr/img/times.txt', 10)



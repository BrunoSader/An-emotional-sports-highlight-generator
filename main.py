import numpy as np
import argparse
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip

from video.scene_detection import detect_scene
from audio.classification import classify_scene, HMMTrainer

filename = 'storage/tmp/match.mp4'
capture = cv2.VideoCapture(filename)
fps = capture.get(cv2.CAP_PROP_FPS)
resizeWidth = 0
video = VideoFileClip(filename)
audio = video.audio
del video
fpf = int(audio.fps/fps) #frames per frame
print(fpf)
i=-1
history = [0]
last = None
frames = []
audioframes = []
scenes = []
while True:
    for chunk in audio.iter_chunks(chunksize=fpf) : #simulates audio stream
        i+=1
        (grabbed, frame) = capture.read()
        audioframes.extend(chunk)
        if not grabbed:
            break

        # Resize frame to width, if specified
        if resizeWidth > 0:
            (height, width) = frame.shape[:2]
            resizeHeight = int(float(resizeWidth / width) * height)
            frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if i > 0 :
            if(detect_scene(frame, last)) :
                ###TODO check ocr if possible
                ###TODO send to classifier
                scene = ImageSequenceClip(frames, fps)
                scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                #scene.write_videofile("storage/tmp/test{}.mp4".format(i))
                scene_class = classify_scene(scene)
                frames = []
                audioframes = []

                # Append only interesting scenes
                if(scene_class == "Crowd" or scene_class == "ExcitedCommentary"):
                    print('intresting scene')
                    scenes.append(scene)
                    history.append(i)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last = frame
scene = ImageSequenceClip(frames, fps)
scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
#scene.write_videofile("storage/tmp/test{}.mp4".format(i))
scene_class = classify_scene(scene)
if(scene_class == "Crowd" or scene_class == "ExcitedCommentary"):
    scenes.append(scene)
    history.append(i)

print(history)

highlight = CompositeVideoClip(scenes)
scene.write_videofile("highlight.mp4")

capture.release()
cv2.destroyAllWindows()
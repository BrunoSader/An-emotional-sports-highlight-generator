import numpy as np
import argparse
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, VideoFileClip
from moviepy.audio.AudioClip import AudioArrayClip
import time
import os

from video.scene_detection import detect_scene
from audio.classification import classify_scene, HMMTrainer
from audio.spectro_analysis import concat_video

filename = 'storage/tmp/matchShort.mkv'
capture = cv2.VideoCapture(filename)
fps = capture.get(cv2.CAP_PROP_FPS)
print(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
scenes_count = 0
start = time.time()

# Delete previous class by second file
if(os.path.isfile("storage/tmp/classBySecond.txt")):
    os.remove("storage/tmp/classBySecond.txt")

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

    last = frame
scene_class = classify_scene(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
if(scene_class == "Crowd" or scene_class == "ExcitedCommentary"):
    scene = ImageSequenceClip(frames, fps)
    scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
    history.append(i)

concat_video(scenes_count, video_path='storage/tmp/scenes/scene', save_path='highlights.mp4')

print(history)

scene.write_videofile("highlight.mp4")
end = time.time()

print("Elapsed time is  {}".format(end-start))

capture.release()
cv2.destroyAllWindows()
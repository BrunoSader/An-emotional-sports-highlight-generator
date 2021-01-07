import numpy as np
import argparse
import cv2
from moviepy.editor import ImageSequenceClip

from video.scene_detection import detect_scene


if __name__ =='__main__' :
    filename = 'storage/tmp/theo_walcott.mp4'
    capture = cv2.VideoCapture(filename)
    fps = capture.get(cv2.CAP_PROP_FPS)
    resizeWidth = 0
    
    i=-1
    history = [0]
    last = None
    frames = []
    while True:
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
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if i > 0 :
            if(detect_scene(frame, last)) :
                ###TODO check ocr if possible
                ###TODO send to classifier
                scene = ImageSequenceClip(frames, fps)
                scene.write_videofile("storage/tmp/test{}.mp4".format(i))
                #classify_scene()
                frames = []
                history.append(i)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        last = frame
    scene = ImageSequenceClip(frames, fps)
    scene.write_videofile("storage/tmp/test{}.mp4".format(i))
    history.append(i)
    #print(history)
    
    capture.release()
    cv2.destroyAllWindows()
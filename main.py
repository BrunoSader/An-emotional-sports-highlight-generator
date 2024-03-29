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
from audio.CNN_classifier import predict
from audio.spectro_analysis import concat_video
# from ocr.final_ocr import ocr
# from ocr.highlights import readFile, highlights
from moviepy.editor import VideoFileClip, concatenate_videoclips

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='CNN', help="CNN, HMM")
parser.add_argument('--OCR', action='store_true')
parser.add_argument('--match-path', type=str, required=True, help="path to match")
parser.add_argument('--fifa', action='store_true')
parser.add_argument('--demo', action='store_true')
parser.add_argument('--algo', action='store_true')
args = parser.parse_args()

filename = args.match_path
capture = cv2.VideoCapture(filename)
fps = capture.get(cv2.CAP_PROP_FPS)
print('chosen model : {}'.format(args.model))
resizeWidth = 0
video = VideoFileClip(filename)
audio = video.audio
fpf = int(audio.fps/fps) #frames per frame
i=-1
history = []
start_indices = [0]
last = None
frames = []
audioframes = []
scenes_count = 0
threshold = 0.95
if args.fifa :
    threshold = 0.95


# Only used with OCR
start_frame = 0
end_frame = 0
scenes = []
video_length = int(video.duration)
del video
print(video_length)

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

if(os.path.isfile("storage/tmp/allScenes.txt")):
    os.remove("storage/tmp/allScenes.txt")

if(os.path.isfile("storage/tmp/prefinalAudioScenes.txt")):
    os.remove("storage/tmp/prefinalAudioScenes.txt")

if(os.path.isfile("storage/tmp/prefinalOCRScenes.txt")):
    os.remove("storage/tmp/prefinalOCRScenes.txt")

if(os.path.isfile("storage/tmp/finalScenes.txt")):
    os.remove("storage/tmp/finalScenes.txt")

f= open("storage/tmp/classByScene.txt","a+")
f1= open("storage/tmp/deletedScenes.txt","a+")
f2= open("storage/tmp/acceptedScenes.txt","a+")
f3= open("storage/tmp/allScenes.txt","a+")
f4= open("storage/tmp/prefinalAudioScenes.txt","a+")
f5= open("storage/tmp/prefinalOCRScenes.txt","a+")
f6= open("storage/tmp/finalScenes.txt","a+")
if(os.path.isdir("storage/tmp/tmpMain/")):
    for filename in os.listdir("storage/tmp/tmpMain/"):
        filepath = os.path.join("storage/tmp/tmpMain/", filename)
        os.remove(filepath)

if args.model == 'CNN' :
    print("Running CNN")
    scene_description = 'Unknown' #used for demo
    for _, chunk in tqdm(enumerate(audio.iter_chunks(chunksize=fpf))) : #simulates audio stream
        i+=1
        end_frame = i
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
            if(detect_scene(frame, last, threshold)) :
                scene_classes = predict(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                d = defaultdict(list)
                for k, v in scene_classes:
                    d[v].append(k)
                l = [(i, np.mean(j)) for i,j in d.items()]
                score = dict(l)
                counter = Counter(elem[1] for elem in scene_classes[-3:])
                unexciting_condition = counter['Unexcited'] >= 2
                if args.fifa :
                    exciting_condition = counter['Excited'] > 0 or counter['Crowd'] > 0
                    crowd_condition = False
                else :
                    exciting_condition = counter['Excited'] > 0 and score['Excited'] > 0.6
                    crowd_condition = counter['Crowd'] > 0 and score['Crowd'] > 0.90
                if not unexciting_condition :
                    if exciting_condition :
                        if len(scene_classes) > 10 :
                            for i, item in enumerate(scene_classes[::-1]):
                                if item[1] == 'Unexcited':
                                    length = i+5
                                    if args.OCR :
                                        print((start_frame/fps, end_frame/fps))
                                        start_frame = end_frame - length
                                    else :
                                        frames = frames[int(-length*fps):]
                                        audioframes = audioframes[int(-length*audio.fps):]
                                    break
                        if args.OCR :
                            print((start_frame/fps, end_frame/fps))
                            scenes.append((start_frame/fps, end_frame/fps))
                        else :
                            scene = ImageSequenceClip(frames, fps)
                            scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                            scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                            scenes_count+=1
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
                print(scene_description)
                print(score)
                start_frame = i

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
    counter = Counter(elem[1] for elem in scene_classes[-3:])
    unexciting_condition = counter['Unexcited'] >= 2
    if args.fifa :
        exciting_condition = counter['Excited'] > 0 or counter['Crowd'] > 0
        crowd_condition = False
    else :
        exciting_condition = counter['Excited'] > 0
        crowd_condition = counter['Crowd'] > 0 and score['Crowd'] > 0.90
    if not unexciting_condition :
        if exciting_condition :
            if len(scene_classes) > 10 :
                for i, item in enumerate(scene_classes[::-1]):
                    if item[1] == 'Unexcited':
                        length = i+5
                        if args.OCR :
                            print((start_frame/fps, end_frame/fps))
                            start_frame = end_frame - length
                        else :
                            frames = frames[int(-length*fps):]
                            audioframes = audioframes[int(-length*audio.fps):]
                        break
            if args.OCR :
                print((start_frame/fps, end_frame/fps))
                scenes.append((start_frame/fps, end_frame/fps))
            else :
                scene = ImageSequenceClip(frames, fps)
                scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
                scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                scenes_count+=1
    frames.clear()
    audioframes.clear()

elif args.model == 'HMM' :

    sceneFiles = []

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
            if(detect_scene(frame, last, threshold)) :
                
                # Write audio & read with audioclip
                clip = AudioArrayClip(np.asarray(audioframes), fps=audio.fps)
                clip.write_audiofile('storage/tmp/tmpMain/sceneTmp' + str(startScene/fps) + '-' + str(endScene/fps) + ' ' + '.wav', audio.fps)
                sceneAudio = AudioFileClip('storage/tmp/tmpMain/sceneTmp' + str(startScene/fps) + '-' + str(endScene/fps) + ' ' + '.wav')
                
                # Getting result action for the scene
                scene_class, classBySec = classify_scene2(sceneAudio, startScene/fps, debug=True)
                sceneFiles.append(["", scene_class, startScene/fps, endScene/fps])

                f.write(str(startScene/fps) + '-' + str(endScene/fps) + ' ' + scene_class + '\n')
                
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
                        count += 4
                    elif(len(classBySec) > count):
                        count += len(classBySec) - count
                    delta = len(classBySec) - count

                    savedFrames = frames[int(delta*fps) : len(frames)]
                    savedAudioFrames = audioframes[int(delta*audio.fps) : len(audioframes)]

                    # Append scene with actual length to response
                    if not args.OCR :
                        scene = ImageSequenceClip(savedFrames, fps)
                        scene = scene.set_audio(AudioArrayClip(np.asarray(savedAudioFrames), fps=audio.fps))
                        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
                    sceneFiles[len(sceneFiles)-1][0] = "storage/tmp/scenes/scene{}.mp4".format(scenes_count)
                    scenes_count+=1
                    history.append(i)
                    start_indices.append(i-len(frames))
                
                f3.write(str(sceneFiles[len(sceneFiles)-1][0]) + ' ' + str(sceneFiles[len(sceneFiles)-1][1]) + ' ' + str(sceneFiles[len(sceneFiles)-1][2]) + ' ' + str(sceneFiles[len(sceneFiles)-1][3]) + '\n')

                frames.clear()
                audioframes.clear()
                startScene = i

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        audioframes.extend(chunk)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        last = frame
        
    scene_class, classBySec = classify_scene2(AudioArrayClip(np.asarray(audioframes), fps=audio.fps), False)
    if(scene_class == "SaveTheEnd" and not args.OCR):
        scene = ImageSequenceClip(frames, fps)
        scene = scene.set_audio(AudioArrayClip(np.asarray(audioframes), fps=audio.fps))
        scene.write_videofile("storage/tmp/scenes/scene{}.mp4".format(scenes_count))
        history.append(i)
        start_indices.append(i-len(frames))

    frames.clear()
    audioframes.clear()

    # Cut isolated-excited-too-short scenes
    excitedSegments = []
    for index in range(len(sceneFiles)):
        # if( index > 1 and index < len(sceneFiles) - 1 and float(sceneFiles[index][3]) - float(sceneFiles[index][2]) <5 and 
        # sceneFiles[index][1] == "SaveTheEnd" and sceneFiles[index-1][1] == "Pass" and sceneFiles[index+1][1] == "Pass"):
            
        #     f1.write( str(sceneFiles[index][0]) + " " + str(sceneFiles[index][1]) + " " +  str(sceneFiles[index][2]) + " " + str(sceneFiles[index][3]) + '\n')
        #     if(os.path.isfile(sceneFiles[index][0])):
        #         os.remove(sceneFiles[index][0])
        # elif(sceneFiles[index][1] == "SaveTheEnd"): #Append to the result array only important scene indices
        #     scenes.append( [sceneFiles[index][2]/fps, sceneFiles[index][3]/fps] )

        if( sceneFiles[index][1] == "SaveTheEnd" ):
            excitedSegments.append(sceneFiles[index])

        elif(sceneFiles[index][1] == "Pass"):
            if(len(excitedSegments) > 0 and  float(excitedSegments[len(excitedSegments) - 1][3]) - float(excitedSegments[0][2]) < 5 ):
                for scene in excitedSegments:
                    f1.write( str(scene[0]) + " " + str(scene[1]) + " " +  str(scene[2]) + " " + str(scene[3]) + '\n')
                    if(os.path.isfile(scene[0])):
                        os.remove(scene[0])
                excitedSegments.clear()
            
            elif(len(excitedSegments) > 0 and  float(excitedSegments[len(excitedSegments) - 1][3]) - float(excitedSegments[0][2]) > 15 ):
                duration = float(excitedSegments[len(excitedSegments) - 1][3]) - float(excitedSegments[0][2])
                count = 0
                while(count + float(excitedSegments[len(excitedSegments) -1 ][3]) - float(excitedSegments[len(excitedSegments) -1 ][2]) < duration/2):
                    scene = excitedSegments.pop()
                    f1.write( str(scene[0]) + " " + str(scene[1]) + " " +  str(scene[2]) + " " + str(scene[3]) + " del in reverse" + '\n')
                    if(os.path.isfile(scene[0])):
                        os.remove(scene[0])
                    count += float(scene[3]) - float(scene[2])
                
                for scene in excitedSegments:
                    scenes.append([scene[2], scene[3]])

                excitedSegments.clear()

            elif( len(excitedSegments) > 0):
                for scene in excitedSegments:
                    scenes.append([scene[2], scene[3]])
                excitedSegments.clear()
    
    for scene in scenes:
        f2.write(str(scene[0]) + ' ' + str(scene[1]) + '\n')



capture.release()
cv2.destroyAllWindows()

if args.OCR :
    ocr('ocr/img', filename, 'times.txt', video_length)
    OCR_scenes = highlights(readFile('ocr/img/times.txt'), 10)
    
    if not args.algo:
        print('OCR',OCR_scenes)
        if len(scenes) > len(OCR_scenes) :
            longer_scenes = scenes
            shorter_scenes = OCR_scenes
        else :
            longer_scenes = OCR_scenes
            shorter_scenes = scenes

        final_scenes = []
        i, j = 0, 0
        while i < len(shorter_scenes) and j < len(longer_scenes) :
            print(shorter_scenes[i],longer_scenes[j])
            if shorter_scenes[i][0] > longer_scenes[j][1] :
                final_scenes.append(longer_scenes[j])
                j+=1
            elif shorter_scenes[i][0] < longer_scenes[j][1] :
                if shorter_scenes[i][0] > longer_scenes[j][0] :
                    shorter_scenes[i] = (longer_scenes[j][0],shorter_scenes[i][1])
                elif shorter_scenes[i][1] < longer_scenes[j][0] :
                    final_scenes.append(shorter_scenes[i])
                    i+=1
                elif shorter_scenes[i][1] < longer_scenes[j][1] :
                    shorter_scenes[i] = (shorter_scenes[i][0],longer_scenes[j][1])
                    j=j+1
                elif shorter_scenes[i][1] > longer_scenes[j][1] :
                    j+=1
            else :
                if shorter_scenes[i][0] > longer_scenes[j][0] :
                    shorter_scenes[i] = (longer_scenes[j][0],shorter_scenes[i][1])
                j+=1
        while i < len(shorter_scenes):
            final_scenes.append(shorter_scenes[i])
            i+=1
        while j < len(longer_scenes):
            final_scenes.append(longer_scenes[j])
            j+=1

        print(final_scenes)

        #generate_highlights_compare('ocr/highlights_videos', 'secondmatch.mkv', 'ocr/img/times.txt', 10, start_indices)
    else:

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

        #         elif(scene[2] == "ocr" and i < len(sortedScenes) - 1 and sortedScenes[i+1][2] == "audio"):
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





        allScenes = []
        for scene in scenes:
            allScenes.append([scene[0], scene[1], "audio"])
        for scene in OCR_scenes:
            allScenes.append([scene[0], scene[1], "ocr"])


        for scene in scenes:
            f4.write( str(scene[0]) + " " + str(scene[1]) + " " + "audio" + '\n')

        for scene in OCR_scenes:
            f5.write( str(scene[0]) + " " + str(scene[1]) + " " + "ocr" + '\n')


        sortedScenes = sorted(allScenes, key=lambda scene: scene[1])

        # Algorithm for choosing the right scenes
        near = 11
        use = True
        final_scenes = []
        for i, scene in enumerate(sortedScenes):
            if(use == True):
                if(scene[2] == "audio" and i < len(sortedScenes) - 1 and sortedScenes[i+1][2] == "ocr"):
                    lenNext = float(sortedScenes[i+1][1]) - float(sortedScenes[i+1][0])
                    lenCurrent = float(scene[1]) - float(scene[0])

                    if(sortedScenes[i+1][0] < scene[0] ):
                        if( lenNext > 40 ):
                            final_scenes.append(scene)
                            use = False
                        else:
                            final_scenes.append(sortedScenes[i+1])

                    elif( sortedScenes[i+1][0] > scene[0] and sortedScenes[i+1][0] <= scene[1] + near):
                        if( lenNext > 40 ):
                            final_scenes.append(scene)
                            use = False
                        else:
                            final_scenes.append(sortedScenes[i+1])
                    
                    else:
                        final_scenes.append(scene)
                    # else:
                    #     continue

                elif(scene[2] == "audio" and i == len(sortedScenes) - 1 and (sortedScenes[i-1][2] == "audio" or sortedScenes[i-1][1] < scene[0] - near)):
                    final_scenes.append(scene)
                
                elif(scene[2] == "audio"):
                    final_scenes.append(scene)

                elif(scene[2] == "ocr" and i < len(sortedScenes) - 1 and sortedScenes[i+1][2] == "audio"):
                    if(sortedScenes[i+1][0] < scene[0]):
                        if( lenCurrent > 40 ):
                            final_scenes.append(sortedScenes[i+1])
                        else:
                            final_scenes.append(scene)
                            use = False

                        final_scenes.append(scene)
                        use = False # TO PUT SOMEWHERE

                    elif( sortedScenes[i+1][0] > scene[0] and sortedScenes[i+1][0] <= scene[1] + near):
                        if( lenCurrent > 40 ):
                            final_scenes.append(sortedScenes[i+1])
                        else:
                            final_scenes.append(scene)
                            use = False
                        
                    else:
                        final_scenes.append(scene)

                elif(scene[2] == "ocr" and i == len(sortedScenes) - 1 and (sortedScenes[i-1][2] == "ocr" or sortedScenes[i-1][1] < scene[0] - near)):
                    final_scenes.append(scene)
                
                elif(scene[2] == "ocr"):
                    final_scenes.append(scene)

            else:
                use = True

        # Delete double cells
        filteredScenes = []
        for scene in final_scenes:
            if(len(filteredScenes) == 0 ):
                filteredScenes.append(scene)
            elif(len(filteredScenes) > 0 and filteredScenes[len(filteredScenes) - 1] != scene):
                filteredScenes.append(scene)

        final_scenes = filteredScenes.copy()
        print(scenes)
        print(final_scenes)

        for scene in final_scenes:
            f6.write( str(scene[0]) + " " + str(scene[1]) + " " + "final" + '\n')

    video = VideoFileClip(filename)
    scenes_count = 0
    for i, chunk in enumerate(final_scenes) :
        print(chunk[0],chunk[1])
        sub = video.subclip(chunk[0], chunk[1])
        sub.write_videofile('storage/tmp/scenes/scene{}.mp4'.format(i))
        scenes_count += 1
    
concat_video(scenes_count, video_path='storage/tmp/scenes/scene', save_path='highlights.mp4')
files = glob.glob('storage/tmp/scenes/*')
for f in files:
    os.remove(f)

end = time.time()

print("Elapsed time is  {}".format(end-start))


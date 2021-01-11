from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips

import pandas as pd


##Global variables
filename='ocr/img/times.txt'
## The length over which an event is considered important
highlight_length = 10
video_path = 'ocr/highlights_videos'
video_name = 'secondmatch.mkv'

def readFile(filename):
    with open(filename) as f:
        list = f.read().splitlines() 
    return list
    
    
def highlights(list):
    highlights = []
    
    for i in range(len(list)-1):
        if((list[i][4] != ',') and (list[i+1][4] != ',')):
            values_i = list[i].split(',')
            values_next = list[i+1].split(',')
            
        if((int(values_next[1])-int(values_i[1]) > highlight_length) and (values_i[0][3]!= '6')):
            highlights.append(values_i[1])
        
    return highlights


def trim_video(video, indices):

    video_index = 0
    for item in indices:
        ##Defining start index and end index for the highlight
        start_index = int(item)-10
        stop_index = int(item)+10
        
        ffmpeg_extract_subclip(video_path +'/'+ video, start_index , stop_index, targetname=video_path+'/' +str(video_index)+'.mkv')
        video_index+=1

    return video_index


def concat_video(video_index):
    videos=[]
    for i in range(video_index):
        videos.append(VideoFileClip(video_path+ '/' + str(i)+'.mkv'))
    
    final_video = concatenate_videoclips(videos)
    final_video.write_videofile(video_path+ '/' +'highlights.mp4')
        

    
if __name__ =='__main__' :
    
    #print(highlights(readFile(filename)))
    #highlights(readFile(filename))
    concat_video(trim_video(video_name, highlights(readFile(filename))))
    #print(video_index)
    #concat_video(2)
    
    

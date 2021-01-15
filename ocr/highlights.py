from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips

import pandas as pd


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
            
        if((int(values_next[1])-int(values_i[1]) > generate_highlights_compare.highlight_length) and (values_i[0][3]!= '6')):
            highlights.append(values_i[1])
        
    return highlights


def trim_video(video, indices):

    video_index = 0
    if indices : 
        for item in indices:
            ##Defining start index and end index for the highlight
                if ((item not in generate_highlights_compare.start_indices)):
                    start_index = int(item)-10
                    stop_index = int(item)+10
                    
                    ffmpeg_extract_subclip(generate_highlights_compare.video_path +'/'+ video, start_index , stop_index, targetname=generate_highlights_compare.video_path+'/' +str(video_index)+'.mkv')
                    video_index+=1

    return video_index


def concat_video(video_index):
    videos=[]
    if video_index > 0 :
        for i in range(video_index):
            videos.append(VideoFileClip(generate_highlights_compare.video_path+ '/' + str(i)+'.mkv'))
        
        final_video = concatenate_videoclips(videos)
        final_video.write_videofile(generate_highlights_compare.video_path+ '/' +'highlights.mp4')
            

def generate_highlights(video_path, video_name, filename, highlight_length):
    generate_highlights.video_path = video_path
    generate_highlights.highlight_length = highlight_length
    concat_video(trim_video(video_name, highlights(readFile(filename))))

def generate_highlights_compare(video_path, video_name, filename, highlight_length, start_indices):
    generate_highlights_compare.video_path = video_path
    generate_highlights_compare.highlight_length = highlight_length
    generate_highlights_compare.start_indices = start_indices
    concat_video(trim_video(video_name, highlights(readFile(filename))))
    
if __name__ =='__main__' :
    filename='ocr/img/times.txt'
    ## The length over which an event is considered important
    highlight_length = 10
    video_path = 'ocr/highlights_videos'
    video_name = 'secondmatch.mkv'
    #print(highlights(readFile(filename)))
    #highlights(readFile(filename))
    generate_highlights(video_path, video_name, filename, highlight_length)
    #print(video_index)
    #concat_video(2)
    
    

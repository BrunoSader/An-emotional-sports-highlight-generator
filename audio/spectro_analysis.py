from preprocessing import transform_to_features, create_feature_average, get_cluster_indices_via_strategy, get_cluster_indices_via_features, create_spectrogram

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import numpy as np


def trim_video(video, indices, min_time):
    video_index = 0
    
    for start_index, stop_index in indices:
        start_index=start_index/40
        stop_index=stop_index/40
        
        if (stop_index - start_index >= min_time) :
            ffmpeg_extract_subclip(video, start_index, stop_index, targetname='storage/trim/video'+str(video_index)+'.mkv')
            video_index+=1
        
    return video_index

def trim_video_final(video, indices):
    video_index = 0
    
    for start_index, stop_index in indices:
        start_index=start_index/40
        stop_index=stop_index/40
        
        if(stop_index - start_index > 5):
            ffmpeg_extract_subclip(video, start_index, stop_index, targetname='storage/trim_final/video'+str(video_index)+'.mkv')
            video_index+=1
        
    return video_index

def concat_video(video_index):
    videos=[]
    for i in range(video_index):
        videos.append(VideoFileClip('storage/trim/video'+str(i)+'.mkv'))
    
    final_video = concatenate_videoclips(videos)
    final_video.write_videofile("storage/tmp/highlights.mp4")
        
def concat_video_final(video_index):
    videos=[]
    for i in range(video_index):
        videos.append(VideoFileClip('storage/trim_final/video'+str(i)+'.mkv'))
    
    final_video = concatenate_videoclips(videos)
    final_video.write_videofile("storage/tmp/highlights_final.mp4")

if __name__ =='__main__' :
    mfcc_feat, fbank_feat = transform_to_features('storage/tmp/audio.wav', mfcc_bool = False, nfilt=5, nfft=4410, winlen=0.2,winstep=0.05)
    mfcc_feat_av, fbank_feat_av = create_feature_average(fbank_feat=fbank_feat, length = 50)
    indices = get_cluster_indices_via_features(fbank_feat_av)
    video_index = trim_video('storage/tmp/highlights.mp4', indices, 4)
    concat_video(video_index)



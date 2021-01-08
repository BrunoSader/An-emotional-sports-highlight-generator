from .preprocessing import transform_to_features, create_feature_average, get_cluster_indices_via_strategy, get_cluster_indices_via_features, create_spectrogram, raw_spectro

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import numpy as np


import librosa
import librosa.display

def trim_video(video, indices, min_time, divider=40):
    video_index = 0
    
    for start_index, stop_index in indices:
        start_index=start_index/divider
        stop_index=stop_index/divider
        
        if (stop_index - start_index >= min_time) :
            ffmpeg_extract_subclip(video, start_index, stop_index, targetname='storage/trim/video'+str(video_index)+'.mkv')
            video_index+=1
        
    return video_index

def concat_video(video_index, video_path='storage/trim/video', save_path = 'storage/tmp/highlights.mp4'):
    videos=[]
    for i in range(video_index):
        videos.append(VideoFileClip(video_path+str(i)+'.mp4'))
    
    final_video = concatenate_videoclips(videos)
    final_video.write_videofile(save_path)

if __name__ =='__main__' :

    mfcc_feat, fbank_feat = transform_to_features('storage/tmp/audio.wav', mfcc_bool = False, nfilt=12, nfft=4410, winlen=0.2,winstep=0.05)
    mfcc_feat_av, fbank_feat_av = create_feature_average(fbank_feat=fbank_feat, length = 80)
    indices = get_cluster_indices_via_features(fbank_feat_av)
    video_index = trim_video('storage/tmp/match.mkv', indices, 4)
    concat_video(video_index)

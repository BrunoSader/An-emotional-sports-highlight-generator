from preprocessing import movingaverage, zero_runs

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from python_speech_features import mfcc, logfbank, get_filterbanks
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.colors as colors

from scipy.cluster.vq import kmeans, vq


def trim_video(video, indices):
    video_index = 0
    
    for start_index, stop_index in indices:
        start_index=start_index/40
        stop_index=stop_index/40
        
        if(stop_index - start_index >= 5):
            ffmpeg_extract_subclip(video, start_index, stop_index, targetname='storage/trim/video'+str(video_index)+'.mkv')
            video_index+=1
        
    return video_index


def concat_video(video_index):
    videos=[]
    for i in range(video_index):
        videos.append(VideoFileClip('storage/trim/video'+str(i)+'.mkv'))
    
    final_video = concatenate_videoclips(videos)
    final_video.write_videofile("storage/tmp/highlights.mp4")
        

if __name__ =='__main__' :
    #ffmpeg_extract_subclip("storage/tmp/audio.wav", 0, 3000, targetname="storage/tmp/res_audio.wav")
    (rate,sig) = wav.read('storage/tmp/audio.wav')
    #mfcc_feat = mfcc(sig,rate, nfft=1103, nfilt=6)
    fbank_feat = logfbank(sig,rate,nfft=4410,nfilt=6,winlen=0.1,winstep=0.05)
    #plt.plot(mfcc_feat)
    #plt.show()
    #plt.plot(fbank_feat)
    #plt.show()

    fbank_feat_av_0 = movingaverage(fbank_feat[:, 0], 80)

    codebook, _ = kmeans(fbank_feat_av_0, 2)  # number of clusters
    cluster_indices, _ = vq(fbank_feat_av_0, codebook)

    ranges = zero_runs(cluster_indices)
    print(len(ranges))
    
    video_index=trim_video('storage/tmp/half_2_highlights.mp4', ranges)
    concat_video(video_index)


    '''
    plt.plot(fbank_feat_av_0)
    plt.plot(cluster_indices)
    plt.show()
    '''
    
    '''fbank_feat_av_0 = movingaverage(fbank_feat[:, 0], 100)
    fbank_feat_av_1 = movingaverage(fbank_feat[:, 1], 100)
    fbank_feat_av_2 = movingaverage(fbank_feat[:, 2], 100)
    fbank_feat_av_3 = movingaverage(fbank_feat[:, 3], 100)
    fbank_feat_av_4 = movingaverage(fbank_feat[:, 4], 100)
    fbank_feat_av_5 = movingaverage(fbank_feat[:, 5], 100)
    plt.plot(fbank_feat_av_1)
    plt.plot(fbank_feat_av_2)
    plt.plot(fbank_feat_av_3)
    plt.plot(fbank_feat_av_4)
    plt.plot(fbank_feat_av_5)
    plt.plot(fbank_feat[:, 0])
    plt.plot(fbank_feat_av_0)
    plt.show()'''

    '''
    firstChannel = sig[:,0]
    f, t, spectro = signal.spectrogram(firstChannel, rate)
    plt.pcolormesh(t, f, spectro, norm=colors.PowerNorm(gamma=0.2))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    '''
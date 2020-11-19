from preprocessing import movingaverage

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from python_speech_features import mfcc, logfbank, get_filterbanks
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans, vq

if __name__ =='__main__' :
    #ffmpeg_extract_subclip("storage/tmp/audio.wav", 0, 3000, targetname="storage/tmp/res_audio.wav")
    (rate,sig) = wav.read('storage/tmp/audio.wav')
    #mfcc_feat = mfcc(sig,rate, nfft=1103, nfilt=6)
    fbank_feat = logfbank(sig,rate,nfft=4410,nfilt=6,winlen=0.1,winstep=0.05)
    #plt.plot(mfcc_feat)
    #plt.show()
    #plt.plot(fbank_feat)
    #plt.show()

    fbank_feat_av_0 = movingaverage(fbank_feat[:, 0], 1000)

    codebook, _ = kmeans(fbank_feat_av_0, 2)  # number of clusters
    cluster_indices, _ = vq(fbank_feat_av_0, codebook)

    plt.plot(fbank_feat_av_0)
    plt.plot(cluster_indices)
    plt.show()
    
    
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

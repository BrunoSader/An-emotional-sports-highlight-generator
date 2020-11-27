from __future__ import division
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
from python_speech_features import mfcc, logfbank, get_filterbanks
import scipy.io.wavfile as wav
from scipy.cluster.vq import kmeans, vq

def __movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def __zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def transform_to_features(audiofile, mfcc_bool=True, fbank_bool=True ,nfft=4410, nfilt=6, winlen=0.1,winstep=0.06):
    (rate,sig) = wav.read(audiofile)
    mfcc_feat = None
    fbank_feat = None
    if mfcc_bool :
        mfcc_feat = mfcc(sig,rate, nfft=nfft, nfilt=nfilt, winlen=winlen,winstep=winstep)
    if fbank_bool :
        fbank_feat = logfbank(sig,rate, nfft=nfft, nfilt=nfilt, winlen=winlen,winstep=winstep)
    return mfcc_feat, fbank_feat

def create_feature_average(mfcc_feat=None, fbank_feat=None, length=100):
    mfcc_feat_av = []
    fbank_feat_av = []
    if mfcc_feat is not None :
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 0], length))
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 1], length))
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 2], length))
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 3], length))
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 4], length))
        mfcc_feat_av.append(__movingaverage(mfcc_feat[:, 5], length))
    if fbank_feat is not None :
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 0], length))
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 1], length))
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 2], length))
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 3], length))
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 4], length))
        fbank_feat_av.append(__movingaverage(fbank_feat[:, 5], length))

    return mfcc_feat_av, fbank_feat_av

### Strategies are "veto", "one-for-all", "democratic"
### "veto" if one feature doesnt agree on the the segment being intreseting it is is removed
### "one-for-all" if one feature agreed on the the segment being intreseting it is is kept
### "democratic" features vote and segment kept by majority 
def get_cluster_indices(feat_arr, strategy='veto'):
    cluster_indices = 0
    if strategy == 'veto' :
        # 1- to invert 0 and 1 (clusters)
        # Multiplication -> inter join
        # 1- to invert back 0 and 1 (clusters)
        codebook, _ = kmeans(feat_arr[0], 2)  # number of clusters
        cluster_indices_0, _ = vq(feat_arr[0], codebook)
        cluster_indices_0 = 1-cluster_indices_0
        codebook, _ = kmeans(feat_arr[1], 2)
        cluster_indices_1, _ = vq(feat_arr[1], codebook)
        cluster_indices_1 = 1-cluster_indices_1
        codebook, _ = kmeans(feat_arr[3], 2)
        cluster_indices_2, _ = vq(feat_arr[3], codebook)
        cluster_indices_2 = 1-cluster_indices_2
        codebook, _ = kmeans(feat_arr[3], 2)
        cluster_indices_3, _ = vq(feat_arr[3], codebook)
        cluster_indices_3 = 1-cluster_indices_3
        codebook, _ = kmeans(feat_arr[4], 2)
        cluster_indices_4, _ = vq(feat_arr[4], codebook)
        cluster_indices_4 = 1-cluster_indices_4
        codebook, _ = kmeans(feat_arr[5], 2)
        cluster_indices_5, _ = vq(feat_arr[5], codebook)
        cluster_indices_5 = 1-cluster_indices_5

        cluster_indices = 1 - cluster_indices_0 * cluster_indices_1 * cluster_indices_2 * cluster_indices_3 * cluster_indices_4 * cluster_indices_5

    elif strategy == 'one-for-all' :
        codebook, _ = kmeans(feat_arr[0], 2)  # number of clusters
        cluster_indices_0, _ = vq(feat_arr[0], codebook)
        cluster_indices_0 = cluster_indices_0
        codebook, _ = kmeans(feat_arr[1], 2)
        cluster_indices_1, _ = vq(feat_arr[1], codebook)
        cluster_indices_1 = cluster_indices_1
        codebook, _ = kmeans(feat_arr[3], 2)
        cluster_indices_2, _ = vq(feat_arr[3], codebook)
        cluster_indices_2 = cluster_indices_2
        codebook, _ = kmeans(feat_arr[3], 2)
        cluster_indices_3, _ = vq(feat_arr[3], codebook)
        cluster_indices_3 = cluster_indices_3
        codebook, _ = kmeans(feat_arr[4], 2)
        cluster_indices_4, _ = vq(feat_arr[4], codebook)
        cluster_indices_4 = cluster_indices_4
        codebook, _ = kmeans(feat_arr[5], 2)
        cluster_indices_5, _ = vq(feat_arr, codebook)
        cluster_indices_5 = cluster_indices_5

        cluster_indices = cluster_indices_0 * cluster_indices_1 * cluster_indices_2 * cluster_indices_3 * cluster_indices_4 * cluster_indices_5

    return(__zero_runs(cluster_indices))

''' 
    ### 1- to invert 0 and 1 (clusters)
    ### Multiplication -> inter join
    ### 1- to invert back 0 and 1 (clusters)
    codebook, _ = kmeans(fbank_feat_av_0, 2)  # number of clusters
    cluster_indices_0, _ = vq(fbank_feat_av_0, codebook)
    cluster_indices_0 = cluster_indices_0
    #codebook, _ = kmeans(fbank_feat_av_1, 2)
    #cluster_indices_1, _ = vq(fbank_feat_av_1, codebook)
    #cluster_indices_1 = 1-cluster_indices_1
    #codebook, _ = kmeans(fbank_feat_av_2, 2)
    #cluster_indices_2, _ = vq(fbank_feat_av_2, codebook)
    #cluster_indices_2 = 1-cluster_indices_2
    #codebook, _ = kmeans(fbank_feat_av_3, 2)
    #cluster_indices_3, _ = vq(fbank_feat_av_3, codebook)
    #cluster_indices_3 = 1-cluster_indices_3
    #codebook, _ = kmeans(fbank_feat_av_4, 2)
    #cluster_indices_4, _ = vq(fbank_feat_av_4, codebook)
    #cluster_indices_4 = 1-cluster_indices_4
    codebook, _ = kmeans(fbank_feat_av_5, 2)
    cluster_indices_5, _ = vq(fbank_feat_av_5, codebook)
    cluster_indices_5 = cluster_indices_5

    cluster_indices = cluster_indices_0 * cluster_indices_5

    plt.plot(cluster_indices)
    plt.show()

    ranges = zero_runs(cluster_indices)
    print(len(ranges))
    
    video_index=trim_video('storage/tmp/match.mkv', ranges)
    concat_video(video_index)


    plt.plot(fbank_feat_av_0)
    plt.plot(cluster_indices)
    plt.show()

    
    fbank_feat_av_0 = movingaverage(fbank_feat[:, 0], 100)
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
    plt.show()

    firstChannel = sig[:,0]
    f, t, spectro = signal.spectrogram(firstChannel, rate)
    plt.pcolormesh(t, f, spectro, norm=colors.PowerNorm(gamma=0.2))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    '''
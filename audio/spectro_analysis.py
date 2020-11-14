import preprocessing

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from python_speech_features import mfcc, logfbank, get_filterbanks
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

if __name__ =='__main__' :
    ffmpeg_extract_subclip("storage/tmp/audio.wav", 0, 3000, targetname="storage/tmp/res_audio.wav")

    (rate,sig) = wav.read('storage/tmp/res_audio.wav')
    mfcc_feat = mfcc(sig,rate, nfft=1103, nfilt=6)
    fbank_feat = logfbank(sig,rate, nfft=4410, nfilt=6, winlen=0.1, winstep=0.05)
    print(len(fbank_feat))
    '''print('#')
    print(len(fbank_feat[0]))
    print('##')
    print(get_filterbanks(nfft=1103))
    print('###')
    print(len(get_filterbanks(nfft=1103)))'''
    #plt.plot(mfcc_feat[:, 0])
    #plt.show()
    plt.plot(fbank_feat)
    plt.show()
#---------------------------
# Spectral Gating Noise Reduction
#---------------------------

import noisereduce as nr
import librosa

if __name__ == '__main__' :

    print('running ...')

    # load data
    data, rate = librosa.load("storage/tmp/audio.wav")
    # select section of data that is noise
    noisy_part, noisy_rate = librosa.load("storage/tmp/crowd_sample.wav")
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)
    librosa.output.write_wav('storage/tmp/reduced_audio.wav', reduced_noise, rate)
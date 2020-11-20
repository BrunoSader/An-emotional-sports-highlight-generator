#---------------------------
# Spectral Gating Noise Reduction
#---------------------------

import noisereduce as nr
# load data
rate, data = wavfile.read("../storage/tmp/audio.wav")
# select section of data that is noise
noisy_part = wavfile.read("../storage/tmp/crowd_sample.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)

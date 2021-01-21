### For training testing and development of the classifier please check :
### speech-classification-using-cnn.ipynb

import numpy as np
import librosa
from keras.models import load_model

classes = ['Crowd', 'Excited', 'Unexcited']
model = load_model('audio/cnn_model.h5', compile=False)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.load_weights("audio/cnn_weights.h5")

def predict(audio):
    tmpfile = 'storage/tmp/tmp.wav'
    audio.write_audiofile(tmpfile)
    samples, sample_rate = librosa.load(tmpfile, sr = 44100)
    chunk_size = sample_rate
    indices = []
    for i in range(0, int(len(samples)/chunk_size)):
        chunk = samples[i*chunk_size:(i+1)*chunk_size]
        prob=model.predict(chunk.reshape(1,sample_rate,1))
        indices.append((prob[0][np.argmax(prob[0])],classes[np.argmax(prob[0])]))
    return indices
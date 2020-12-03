import noisereduce as nr
import librosa
import soundfile as sf
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

# create a speech recognition object
r = sr.Recognizer()

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 1000,
        # adjust this per requirement
        silence_thresh = sound.dBFS-7,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "storage/tmp/audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened, language = "fr-FR")
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

if __name__ == '__main__' :

    ## Spectral Gating Noise Reduction
    print('running ...')

    ## load data
    #data, rate = librosa.load("storage/tmp/audio.wav")
    ## select section of data that is noise
    #noisy_part, noisy_rate = librosa.load("storage/tmp/crowd_sample.wav")
    ## perform noise reduction
    #reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False)
    #sf.write('storage/tmp/filtered_audio.wav', reduced_noise, rate)

    ## Speech to text
    #path = "storage/tmp/filtered_audio.wav"
    #print("\nFull text:", get_large_audio_transcription(path))
    print("\nFull text:", get_large_audio_transcription("storage/tmp/audio.wav"))

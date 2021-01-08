# An-emotional-sports-highlight-generator
Sport Highlight Generator project led by [CAVAGNA Margaux](), [ABDEL RAHMAN Ahmed](), [CINCUIC Robert]() and [SADER Bruno]() in collaboration with Ignifai, Liris and Icar.

# Audio classification:
- Classes used: crowd, excited commentary, unexcited commentary
- Algorithm:
  - Training steps:
    - Train a hidden markov model on the training audio set based on the mfcc features
    - Export the model weights to a pickle file for further use
  
  - Main execution:
    - Load the hidden markov model from the pickle file
    - Applying a shifting time window (of small length) over the target audio (audio of a scene)
    - The step of the shift is 1 second
    - Compute the class for the window with the trained hidden markov model
    - Apply this information to all seconds of the current window
    - Shift window + repeat steps 4 & 5 until the end of the audio
    - Each second has a class distribution (number of times a class was associated to it)
    - Finally associate each second with the class having the highest count on that second
    - Return the dominant class //TODO : change the return type 

- Description of HMMTrainer class:
  - __init__ (self, model_name='GaussianHMM', n_components=10, cov_type='diag', n_iter=1000): initializes a Gaussian hidden markov model
  - train(self, X): fits the HMM with the training set X
  - get_score(self, input_data): returns the score for each class based on the input audio

- Description of functions:
  - getHmmModel(input_folder): returns a hidden markov model trained on the audio files inside the input_folder (each subfolder represents a class)
  
  - classify_scene(videoScene): returns the dominant class label in that scene
    - Represents the execution of the entire pipeline
    - Calls the rest of the functions described below
    
  - getAudioWindows(audio, segmLength): returns a list of audio segments of length segmLength from the input audio
  
  - getClassesInitWin(sampleRate, audioWindows, hmm_models): returns for each audioWindow (for all seconds inside) the class label
  
  - getClassDistrib(windowsClasses, segmLength): returns a list of seconds and their distribution of classes
  
  - labelEachSec(distribBySec): returns a list of seconds and their dominant class label
  
  - concatIntoSegments(classBySec): returns a list of segments(segmentStart, segmentEnd, classLabel)
    - Seconds that have identical dominant class labels are concatenated together 

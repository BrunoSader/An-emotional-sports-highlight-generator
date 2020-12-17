import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import face_recognition
from keras.models import load_model
from keras.preprocessing import image

if __name__ =='__main__' :
    filename = 'storage/tmp/goal_coutinho.mp4'
    capture = cv2.VideoCapture(filename)
    color = 'rgb'
    bins = 16
    resizeWidth = 0

    emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    label_map = dict((v,k) for k,v in emotion_dict.items()) 

    fig, ax = plt.subplots()
    if color == 'rgb':
        ax.set_title('Histogram (RGB)')
    else:
        ax.set_title('Histogram (grayscale)')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Frequency')

    lw = 3
    alpha = 0.5
    if color == 'rgb':
        lineR, = ax.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha)
        lineG, = ax.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha)
        lineB, = ax.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha)
    else:
        lineGray, = ax.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw)
    ax.set_xlim(0, bins-1)
    ax.set_ylim(0, 1)
    plt.ion()
    plt.show()

    #Grab, process, and display video frames. Update plot line object(s).
    i=-1
    history = []
    while True:
        i+=1
        (grabbed, frame) = capture.read()

        if not grabbed:
            break

        # Resize frame to width, if specified.
        if resizeWidth > 0:
            (height, width) = frame.shape[:2]
            resizeHeight = int(float(resizeWidth / width) * height)
            frame = cv2.resize(frame, (resizeWidth, resizeHeight),
                interpolation=cv2.INTER_AREA)

        # Normalize histograms based on number of pixels per frame.
        numPixels = np.prod(frame.shape[:2])
        if color == 'rgb':
            cv2.imshow('RGB', frame)
            (b, g, r) = cv2.split(frame)
            histogramR = cv2.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
            histogramG = cv2.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
            histogramB = cv2.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
            lineR.set_ydata(histogramR)
            lineG.set_ydata(histogramG)
            lineB.set_ydata(histogramB)
            histogram = [histogramR, histogramG, histogramR]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Grayscale', gray)
            histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
            lineGray.set_ydata(histogram)
        history.append(np.float32(histogram))
        if i > 0 :
            if cv2.compareHist(history[-1], history[-2], cv2.HISTCMP_CORREL) < 0.95 :
                print("diff")
            face_locations = face_recognition.face_locations(frame)
            print(face_locations)
            if len(face_locations) > 0 :
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                cv2.imshow('face', face_image)
                cv2.waitKey(0)
                face_image = cv2.resize(face_image, (48,48))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
                model = load_model("video/emotion_detector_models/model_v6_23.hdf5")
                predicted_class = np.argmax(model.predict(face_image))
                predicted_label = label_map[predicted_class]
                print(predicted_label)
        fig.canvas.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    capture.release()
    cv2.destroyAllWindows()
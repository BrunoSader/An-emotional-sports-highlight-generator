import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

if __name__ =='__main__' :
    filename = 'storage/tmp/highlights.mp4'
    capture = cv2.VideoCapture(filename)
    color = 'rgb'
    bins = 16
    resizeWidth = 0

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
    while True:
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
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Grayscale', gray)
            histogram = cv2.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
            lineGray.set_ydata(histogram)
        fig.canvas.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
import cv2
import numpy as np


path = "/mnt/c/Users/Harshul/Downloads/dictator.mp4"
cap = cv2.VideoCapture(path)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, buf[fc] = cap.read()
    fc += 1

cap.release()


cv2.waitKey(0)
np.save("/mnt/c/Users/Harshul/Desktop/dict.npy", buf)
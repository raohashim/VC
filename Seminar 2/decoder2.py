import cv2
import numpy as np
import pickle
import scipy.signal as sp
import functions as func

cv2.namedWindow('Reconstructed RGB with Pyramid Filter', cv2.WINDOW_NORMAL)

N = 2
#Creating the Rectangular filters
rect = np.ones((N,N))/N
#Creating the Pyramid filters
pyra = sp.convolve2d(rect,rect)/N

reducedF = open('videorecord.txt', 'rb')
try:
    ctr = 1
    while True:
        Y = pickle.load(reducedF)
        Y = Y.astype(np.uint8)
        r, c = Y.shape
	downCb = pickle.load(reducedF)
        downCr = pickle.load(reducedF)
	#Up sampling of Cb and Cr
        sampledCb = func.sampler420(downCb)
        sampledCr = func.sampler420(downCr)
        pyraFY = Y
        pyraFCb = sp.convolve2d(sampledCb, pyra, mode='same')
        pyraFCr = sp.convolve2d(sampledCr, pyra, mode='same')     
	pyraFrame = np.zeros((r, c, 3))
        pyraFrame[:, :, 0] = Y
        pyraFrame[:, :, 1] = pyraFCr
        pyraFrame[:, :, 2] = pyraFCb
	pyraFrame = func.ycbcr2rgb(pyraFrame)
    
        cv2.imshow('Reconstructed RGB with Pyramid Filter', pyraFrame)

        if cv2.waitKey(400) & 0xFF == ord('q'):
            break
except (EOFError):
    pass

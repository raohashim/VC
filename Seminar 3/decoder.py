import cv2
import numpy as np
import pickle
import scipy.signal as sp
import os
import functions


dctt = open('dct.txt', 'rb')
xf = open('full.txt', 'rb')

try:
    while True:

        dctReduced = pickle.load(dctt)
        fullFrame = pickle.load(xf)

        dct = functions.DCTWithZeros(dctReduced, functions.DTCFactor)

        idctFrame = functions.IDCTFrame(dct)

        frame_sub = functions.chomaSubSamp(idctFrame[:,:,0], idctFrame[:,:,1], idctFrame[:,:,2])


        if functions.filterON:
            frame_filt = functions.filterFrame(frame_sub, functions.desiredFilter)
        else:
            frame_filt = frame_sub
        RGB_frame = functions.Ycbcr2Rgb(frame_filt)

        cv2.imshow('RGB frame', RGB_frame)
        if cv2.waitKey(200) == ord('q'):
            break

except (EOFError):
    pass

dctt.close()
xf.close()
cv2.destroyAllWindows()


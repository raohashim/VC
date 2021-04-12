import numpy as np        
import cv2
import sys
import cPickle as pickle

f=open('video_raw_data.txt', 'r')

while(True):

    reduced=pickle.load(f)
    framedec=reduced.copy() 
    framedec = framedec.astype(np.float)
    Y,Cb,Cr = framedec[:,:,0],framedec[:,:,1],framedec[:,:,2]     
    #convert YCbCr to rgb
    Cb = Cb-128
    Cr = Cr-128
    r = (Y+Cr*1.4025)
    g = (Y+Cb*(-0.34434)+Cr*(-0.7144))
    b = (Y+Cb*1.7731)
    framedec[:,:,0],framedec[:,:,1],framedec[:,:,2] = b,g,r
    cv2.imshow('Video1',framedec/255)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

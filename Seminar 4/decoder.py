import numpy as np        
import cv2
import sys
import cPickle as pickle
import scipy.signal

f=open('videorecord.txt', 'r')

rows,cols = 480,640
n=0
N = 2
frame = np.zeros((480,640,3))
frame1 = np.zeros((480,640,3))
sampledCb = np.zeros((rows,cols))           #subsample Matrix
sampledCr = np.zeros((rows,cols))

filt1=np.ones((N,N))/N;
filt2=scipy.signal.convolve2d(filt1,filt1)/N

videolist=np.load(f)

fftMr=np.ones((rows,1))                               # FFT matrix
fftMr[int(rows/8.0):int(rows-rows/8.0),0]=np.zeros(int(3.0/4.0*rows))
fftMc=np.ones((1,cols)) 
fftMc[0,int(cols/8.0):int(cols-cols/8.0)]=np.zeros(int(3.0/4.0*cols));
fftM=np.dot(fftMr,fftMc)

while(True):

    Y,Cb,Cr = videolist[n*3],videolist[n*3+1],videolist[n*3+2]   #convert YCbCr to rgb
    n = n+1;
    sampledCb[0::N,0::N] = Cb
    sampledCr[0::N,0::N] = Cr
    Cb=scipy.signal.convolve2d(sampledCb,filt1,mode='same')
    Cr=scipy.signal.convolve2d(sampledCr,filt1,mode='same')


    #cv2.imshow('Y',Y)
    #cv2.imshow('cb',Cb)
    #cv2.imshow('cr',Cr)

    Y = Y*255
    Cb = Cb*255-128
    Cr = Cr*255-128

    r = (Y+Cr*1.4025)
    g = (Y+Cb*(-0.34434)+Cr*(-0.7144))
    b = (Y+Cb*1.7731)

    frame[:,:,0],frame[:,:,1],frame[:,:,2] = b,g,r

    cv2.imshow('Video',frame/255)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

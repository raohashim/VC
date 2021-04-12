import numpy as np        
import cv2
import sys
import cPickle as pickle
import scipy.signal
import scipy.fftpack as sft

f=open('videorecord.txt', 'r')

rows,cols = 480,640
n=0
N = 2
frame = np.zeros((480,640,3))
frame1 = np.zeros((480,640,3))

filt1=np.ones((N,N))/N;
filt2=scipy.signal.convolve2d(filt1,filt1)/N

sampledCb = np.zeros((rows,cols))           #subsample Matrix
sampledCr = np.zeros((rows,cols))

videolist=np.load(f)

fftMr=np.ones((rows,1))                               # FFT matrix
fftMr[int(rows/8.0):int(rows-rows/8.0),0]=np.zeros(int(3.0/4.0*rows))
fftMc=np.ones((1,cols)) 
fftMc[0,int(cols/8.0):int(cols-cols/8.0)]=np.zeros(int(3.0/4.0*cols));
fftM=np.dot(fftMr,fftMc)

Y = np.zeros((rows,cols))
Ypre = np.zeros((rows,cols))
block=np.array([8,8])

while(True):

    if n%2 == 0:
         print("Frame no",n)
         Y = np.zeros((rows,cols)) 
         Y[0::8,0::8] = videolist[n]
         #Y,Cb,Cr = videolist[n*3],videolist[n*3+1],videolist[n*3+2]   #convert YCbCr to rgb
         n = n+1;

         dctY=np.reshape(Y,(-1,8), order='C')          #reshape n*8
         dctY=sft.idct(dctY,axis=1,norm='ortho')          #dct
         dctY=np.reshape(dctY,(-1,cols), order='C')          #reshape back
         dctY=np.reshape(dctY.T,(-1,8), order='C')        #reshape.T n*8
         dctY=sft.idct(dctY,axis=1,norm='ortho')          #dct
         Y=(np.reshape(dctY,(-1,rows), order='C')).T       #reshape back

         Ypre = Y.copy()

         cv2.imshow('Video',Y[120:360,160:480])

    else:
         print("Frame no",n)
         mv = videolist[n].astype(int)
         n = n+1;

         for yblock in range(30):
            block[0]=yblock*8+120;
            for xblock in range(40):
               block[1]=xblock*8+160;
               Y = Ypre
               Y[block[0]:block[0]+8,block[1]:block[1]+8] = Ypre[block[0]+mv[yblock,xblock,0]:block[0]+mv[yblock,xblock,0]+8,block[1]+mv[yblock,xblock,1]:block[1]+mv[yblock,xblock,1]+8]

         cv2.imshow('Video',Y[120:360,160:480])
    #sampledCb[0::N,0::N] = Cb
    #sampledCr[0::N,0::N] = Cr

    #Cb=scipy.signal.convolve2d(sampledCb,filt1,mode='same')
    #Cr=scipy.signal.convolve2d(sampledCr,filt1,mode='same')

    #cv2.imshow('Y',Y)
    #cv2.imshow('cb',Cb)
    #cv2.imshow('cr',Cr)

    #Y = Y*255
    #Cb = Cb*255-128
    #Cr = Cr*255-128

    #r = (Y+Cr*1.4025)
    #g = (Y+Cb*(-0.34434)+Cr*(-0.7144))
    #b = (Y+Cb*1.7731)

    #frame[:,:,0],frame[:,:,1],frame[:,:,2] = b,g,r

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

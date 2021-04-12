import numpy as np
import cv2
import pickle 
import scipy.signal

cap = cv2.VideoCapture(0)

f=open('videorecord.txt', 'wb')

framenum = 200
N = 2      #subsample value
data = []

[ret, frame] = cap.read()
[rows,cols,c]=frame.shape;

filt1=np.ones((N,N))/N;
filt2=scipy.signal.fftconvolve(filt1,filt1)/N

Yprev=np.zeros((rows,cols))
framevectors=np.zeros((rows,cols,3))
mv=np.zeros((int(rows/8),int(cols/8),2))

for n in range(framenum):
#while(True):
    
    print("Frame no",n)
    [ret, frame] = cap.read()

    if ret==True:

        b,g,r = frame[:,:,0],frame[:,:,1],frame[:,:,2]   #convert rgb to YCbCr

        Y = (r*0.299+g*0.587+b*0.114)/255;   #0~255
        Cb = ((r*(-0.16864)+g*(-0.33107)+b*0.4997)+128)/255;  #0~255
        Cr = ((r*0.499813+g*(-0.418531)+b*(-0.081282))+128)/255;  #0~255


        cv2.imshow('Original',frame/255.0+framevectors)

        filtCr=scipy.signal.fftconvolve(Cr,filt2,mode='same')
        filtCb=scipy.signal.fftconvolve(Cb,filt2,mode='same')

        sampledCb = filtCb[0::N,0::N]      #4:2:0
        sampledCr = filtCr[0::N,0::N]

        block=np.array([8,8])
        framevectors=np.zeros((rows,cols,3))
       
        for yblock in range(30):
           block[0]=yblock*8+200;
           for xblock in range(40):
              block[1]=xblock*8+300;
              Yc=Y[block[0]:block[0]+8 ,block[1]:block[1]+8]   #current block
	      Yp=Yprev[block[0]-4 : block[0]+12 ,block[1]-4 : block[1]+12]  #previous block
              Ycorr=scipy.signal.correlate2d(Yp, Yc,mode='valid')
              index1d=np.argmax(Ycorr)
              index2d=np.unravel_index(index1d,(9,9))
              mv[yblock,xblock]=np.subtract(index2d,(4,4))  
              if sum(mv[yblock,xblock]) != 0:              
                 cv2.line(framevectors, (block[1], block[0]), (block[1]+mv[yblock,yblock,1].astype(int),block[0]+mv[yblock,yblock,0].astype(int)) , (1.0,1.0,1.0));
        Yprev=Y.copy();

        data.append(Y)
        data.append(sampledCb)
        data.append(sampledCr)

        key=cv2.waitKey(1) & 0xFF;
        if key == ord('q'):
           break
    else:
        break

np.save(f,data)

cap.release()
f.close()
cv2.destroyAllWindows()

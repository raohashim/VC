import numpy as np
import cv2
import pickle 
import scipy.signal
import scipy.fftpack as sft

cap = cv2.VideoCapture(0)

f=open('videorecord.txt', 'wb')
f2=open('videotest.txt', 'wb')

framenum = 30
N = 2      #subsample value
M = 1      #dct factor

data = []
data2 = []

[ret, frame] = cap.read()
[rows,cols,c]=frame.shape;
#rows,cols = 480,640

filt1=np.ones((N,N))/N;
filt2=scipy.signal.fftconvolve(filt1,filt1)/N

Yprev=np.zeros((rows,cols))
framevectors=np.zeros((rows,cols,3))
mv=np.zeros((30,40,2))

dctMr=np.ones(8)                                   # DCT matrix
dctMr[M:]=np.zeros(8-M)
dctMc=dctMr;

for n in range(framenum):
#while(True):
    
    print("Frame no",n)
    [ret, frame] = cap.read()

    if ret==True:

        b,g,r = frame[:,:,0],frame[:,:,1],frame[:,:,2]   #convert rgb to YCbCr

        Y = (r*0.299+g*0.587+b*0.114)/255;   #0~255
        #Cb = ((r*(-0.16864)+g*(-0.33107)+b*0.4997)+128)/255;  #0~255
        #Cr = ((r*0.499813+g*(-0.418531)+b*(-0.081282))+128)/255;  #0~255
        #Cb = (r*(-0.16864)+g*(-0.33107)+b*0.4997)/255  #-127~127
        #Cr = (r*0.499813+g*(-0.418531)+b*(-0.081282))/255  #-127~127

        #cv2.imshow('Original',frame/255+framevectors)
        cv2.imshow('original',frame)

        #filtCr=scipy.signal.fftconvolve(Cr,filt2,mode='same')
        #filtCb=scipy.signal.fftconvolve(Cb,filt2,mode='same')

        #sampledCb = filtCb[0::N,0::N]      #4:2:0
        #sampledCr = filtCr[0::N,0::N]

        if n%2 == 0:

          dctY=np.reshape(Y,(-1,8), order='C')             #reshape n*8
          dctY=sft.dct(dctY,axis=1,norm='ortho')           #dct
          dctY=np.dot(dctY,np.diag(dctMr))
          dctY=np.reshape(dctY,(-1,cols), order='C')       #reshape back
          dctY=np.reshape(dctY.T,(-1,8), order='C')        #reshape.T n*8
          dctY=sft.dct(dctY,axis=1,norm='ortho')           #dct
          dctY=np.dot(dctY,np.diag(dctMc))
          dctY=(np.reshape(dctY,(-1,rows), order='C')).T   #reshape back  

          s = dctY[0::8,0::8]
          print("dct",s.shape)
          data.append(s)
          data2.append(s)

        else:

          block=np.array([8,8])
          framevectors=np.zeros((rows,cols,3))
       
          for yblock in range(30):
             block[0]=yblock*8+120;
             for xblock in range(40):
                block[1]=xblock*8+160;
                Yc=Y[block[0]:block[0]+8 ,block[1]:block[1]+8]   #current block
                Yp=Yprev[block[0]-4 : block[0]+12 ,block[1]-4 : block[1]+12]  #previous block
                #bestmae=10000.0;
                #for ymv in range(-8,8):
                #    for xmv in range(-8,8):
                #       diff=Yc-Yprev[block[0]+ymv:block[0]+ymv+8, block[1]+xmv:block[1]+xmv+8];
                #       mae=sum(sum(np.abs(diff)))/64;
                #       if mae< bestmae:
                #         bestmae=mae;
                #         mv[yblock,xblock,0]=ymv;
                #         mv[yblock,xblock,1]=xmv;

                Ycorr=scipy.signal.correlate2d(Yc, Yp,mode='valid')
                #print(Ycorr)
                index1d=np.argmax(Ycorr)
                index2d=np.unravel_index(index1d,(9,9))
                mv[yblock,xblock]=np.subtract(index2d,(4,4))  
                if Ycorr[4,4] == Ycorr[index2d[0],index2d[1]]:
                   mv[yblock,xblock] = np.array([0,0])
                if sum(np.abs(mv[yblock,xblock])) < 4:    
                   mv[yblock,xblock]=np.array([0,0])
                else:
                   cv2.line(framevectors, (block[1], block[0]), (block[1]+mv[yblock,yblock,1].astype(int),block[0]+mv[yblock,yblock,0].astype(int)) , (1.0,1.0,1.0));

          print("mv",mv.shape)
          data.append(mv)
          data2.append(dctY[0::8,0::8])

        Yprev=Y.copy();

        #data.append(sampledCb)
        #data.append(sampledCr)

        key=cv2.waitKey(1) & 0xFF;
        if key == ord('q'):
           break
    else:
        break

np.save(f,data)
np.save(f2,data2)

cap.release()
f.close()
f2.close()
cv2.destroyAllWindows()

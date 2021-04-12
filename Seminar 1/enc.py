import numpy as np
import cv2
import pickle 

cap = cv2.VideoCapture(0)

f=open('video_raw_data.txt', 'wb')
for n in range(300):

    ret, frame = cap.read()
    
    if ret==True:

        cv2.imshow('frame',frame)
        b,g,r = frame[:,:,0],frame[:,:,1],frame[:,:,2]   
	#convert rgb to YCbCr
        Y = (r*0.299+g*0.587+b*0.114)   #0~255
        Cb = ((r*(-0.16864)+g*(-0.33107)+b*0.4997)+128)  #0~255
        Cr = ((r*0.499813+g*(-0.418531)+b*(-0.081282))+128)  #0~255
        cv2.imshow('Y',Y/255)
        cv2.imshow('Cb',Cb/255)
        cv2.imshow('Cr',Cr/255)   
        frameenc = frame.copy()
        frameenc[:,:,0],frameenc[:,:,1],frameenc[:,:,2]= Y,Cb,Cr
        reduced=np.array(frameenc,dtype='uint8')
        pickle.dump(reduced,f)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
f.close()
cv2.destroyAllWindows()

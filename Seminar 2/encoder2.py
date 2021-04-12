import cv2
import numpy as np
import pickle
import scipy.signal
import functions as func
#Two files Created
reducedF = open('videorecord.txt', 'wb')
original = open('Original.txt', 'wb')

N = 2
#Rectangular and Pyramid Filter Created
rect = np.ones((N,N))/N
pyra = scipy.signal.convolve2d(rect,rect)/N

cap = cv2.VideoCapture(0)
for i in range(300):
    ret, frame = cap.read()
    if ret == True:
        R = frame[:, :, 2]
        G = frame[:, :, 1]
        B = frame[:, :, 0]
	#Conversion to YCbCr
	Y,Cb,Cr = func.rgb2ycbcr(frame)
	originalF = np.zeros(frame.shape, np.int8)
        originalF[:, :, 0] = Y
        originalF[:, :, 1] = Cr
        originalF[:, :, 2] = Cb
                     
	#Applying Rectangular Filter to Cb and Cr
        CbRec = scipy.signal.convolve2d(Cb,rect,mode='same')
        CrRec = scipy.signal.convolve2d(Cr,rect,mode='same')
	#Applying Pyramid Filter to Cb and Cr
        CbPyr = scipy.signal.convolve2d(Cb, pyra, mode='same')
        CrPyr = scipy.signal.convolve2d(Cr, pyra, mode='same')
        # Chroma Subsampling with zeros 4:2:0
        DCb = np.zeros(Cb.shape)
        DCb[0::N, 0::N] = Cb[0::N, 0::N]
        DCr = np.zeros(Cr.shape)
        DCr[0::N, 0::N] = Cr[0::N, 0::N]
	
	#Downasmpled Cb and Cr show
        cv2.imshow('Cb Downsampled 4:2:0', (DCb / 255))
        cv2.imshow('Cr Downampled 4:2:0', (DCr / 255))
	#Show results after applying Rectangular and Pyramid Filter	
	cv2.imshow('Rectangle Filter Cb ', CbRec/255)
        cv2.imshow('Rectangle Filter Cr ', CrRec/255)
        cv2.imshow('Pyramid Filter Cb', CbPyr / 255)
        cv2.imshow('Pyramid Filter Cr ', CrPyr / 255)
	
	#Save Orignal Credential to File
	pickle.dump(originalF, original)
        
	#For Saving Pyramid Filter Credentials to file
	reduced = frame.copy()
	enc = np.zeros(reduced.shape,dtype='int8')
        # enc = np.array(reduced, dtype='uint8')
        enc[:, :, 0] = Y
        enc[:, :, 1] = CrPyr
        enc[:, :, 2] = CbPyr
        # Chroma Downsampling 
        pickle.dump(enc[:, :, 0], reducedF)
        pickle.dump(enc[0::2, 0::2, 1], reducedF)
        pickle.dump(enc[0::2, 0::2, 2], reducedF)

        if cv2.waitKey( 1 ) & 0xFF == ord('q'):
            break
#Get the file size for reduced and Orignal
a = reducedF.tell() / 1024.0 ** 2
b = original.tell() / 1024.0 ** 2

print('File size of the actual image ')
print(b, 'Mb')
print('File size of the reduced Image')
print(a, 'Mb')
print('Compression factor')
print(b / a )  # Uncompressed data/ Compressed data
print('Space saving')
print((1 - (a / b)) * 100.0, 'percent')

cap.release()
cv2.destroyAllWindows()
original.close()
reducedF.close()

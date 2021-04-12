import cv2
import numpy as np
import pickle
import scipy.signal as sp
import os
import functions

f_downsampled = open('subs.txt', 'wb')
f_full = open('full.txt', 'wb')
encDCT = open('dct.txt', 'wb')

cap = cv2.VideoCapture(0)

#while(True):
for i in range(25):
    ret, frame = cap.read()
    reduced = frame.copy()

    if functions.filterON:
        frameFilt = functions.filterFrame(functions.Rgb2Ycbcr(reduced), functions.desiredFilter)
    else:
        frameFilt = functions.Rgb2Ycbcr(reduced)

    Y, Cb, Cr = frameFilt[:, :, 0], frameFilt[:, :, 1], frameFilt[:, :, 2]
    cv2.imshow('Y Frame', Y/255.0)
    #Chroma Downsampling with zeros

    Cb_downsampled = np.zeros(Cb.shape)
    Cb_downsampled[0::2, 0::2] = Cb[0::2, 0::2]
    Cr_downsampled = np.zeros(Cr.shape)
    Cr_downsampled[0::2, 0::2] = Cr[0::2, 0::2]

    #cv2.imshow('DCb Frame', Cb_downsampled / 255.0)
    #cv2.imshow('DCr Frame', Cr_downsampled / 255.0)
    #cv2.imshow('Original', reduced)
    
    frame_subsample = np.zeros((reduced.shape))
    frame_subsample[:,:,0] = Y
    frame_subsample[:,:,1] = Cb_downsampled
    frame_subsample[:,:,2] = Cr_downsampled
    
    ssycbcr_wo_zeros = np.zeros((240,320,2)) 
    ssycbcr_wo_zeros[0::, 0::, 0]=frame_subsample[0::2, 0::2, 1]
    ssycbcr_wo_zeros[0::, 0::, 1]=frame_subsample[0::2, 0::2, 2]
        	
    ssycbcr_wo_zeros = functions.DCTRemoveZeros(frame_subsample, functions.DTCFactor)
        
    #frame_subsample = np.zeros((reduced.shape))
    #frame_subsample[:,:,0] = Y
    #frame_subsample[:,:,1] = Cb_downsampled
    #frame_subsample[:,:,2] = Cr_downsampled 

    dcty = functions.applyDCT(Y)
    dctcb = functions.applyDCT(ssycbcr_wo_zeros[::, ::, 0])
    dctcr = functions.applyDCT(ssycbcr_wo_zeros[::, ::, 1])
    
    cv2.imshow('Y DCT', np.abs(dcty[:,:,0]))
    cv2.imshow('Cr DCT', np.abs(dctcb[:, :, 1]))
    cv2.imshow('Cb DCT', np.abs(dctcr[:, :, 2]))

    #dct_wo_Zeros = functions.DCTRemoveZeros(dct, functions.DTCFactor)
    
    enc = np.zeros(frame.shape, np.int8)
    enc[:, :, 0] = Y
    enc[:, :, 1] = Cr_downsampled
    enc[:, :, 2] = Cb_downsampled

    # Storing the data    
    pickle.dump(enc, f_full)

    # Zero removed downsampled version
    pickle.dump(Y, f_downsampled)
    pickle.dump(ssycbcr_wo_zeros[::, ::, 0], f_downsampled)
    pickle.dump(ssycbcr_wo_zeros[::, ::, 1], f_downsampled)

    # Storing DCT data
    #pickle.dump(dct_wo_Zeros.astype(np.float16), encDCT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Size of encoded DCT data: MB", format(encDCT.tell()/1024**2.))
print("original file:   MB",format(f_full.tell()/1024**2.))
print("compression ratio", (f_full.tell())/encDCT.tell())


f_downsampled.close()
f_full.close()
encDCT.close()
cap.release()
cv2.destroyAllWindows()

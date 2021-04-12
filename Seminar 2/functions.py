import numpy as np
import scipy.signal as sp
import math

def sampler420(frame):
    """
    frame: 2D component(for example Cb or Cr)
    N:     Upsampling factor for 420 scheme.
           Should be selected based on Downsampling factor
    """
    r, c = frame.shape
    sampleFrame = np.zeros((r*2,c*2))
    sampleFrame[::2, ::2] = frame
    sampleFrame[1::2,] = sampleFrame[::2,]
    sampleFrame[:,1::2] = sampleFrame[:,::2]
    return sampleFrame

def upsample(frame):

    N = 2
    r, c = frame.shape
    upFrame = np.zeros((r*N,c*N))
    upFrame[::N, ::N] = frame

    return upFrame

def rgb2ycbcr(frame):

    R = frame[:,:,2]
    G = frame[:,:,1]
    B = frame[:,:,0]

    #red = frame.copy()
    Y = (0.299*R + 0.587*G + 0.114*B)
    Cb = (-0.16864*R - 0.33107*G + 0.49970*B)
    Cr = (0.499813*R - 0.418531*G - 0.081282*B)

    return Y, Cb, Cr

def ycbcr2rgb(frame):

    Y = (frame[:,:,0])/255.
    Cb = (frame[:,:,1])/255.
    Cr = (frame[:,:,2])/255.

    '''Compute RGB components'''
    R = (0.771996*Y -0.404257*Cb + 1.4025*Cr)
    G = (1.11613*Y - 0.138425*Cb - 0.7144*Cr)
    B = (1.0*Y + 1.7731*Cb)

    '''Display RGB Components'''
    RGBFrame = np.zeros(frame.shape)
    RGBFrame[:,:,2] = R
    RGBFrame[:,:,1] = G
    RGBFrame[:,:,0] = B

    return RGBFrame


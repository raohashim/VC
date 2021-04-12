import numpy as np
import scipy.signal as sp
import scipy.fftpack as sft
import cv2

N = 2
lpF = np.ones((N, N))/N                     #Rectangular filter
pyF = sp.convolve2d(lpF, lpF)/N             #Pyramidal filter

desiredFilter = pyF                 #Filter selection either lpF and pyF
filterON = True                     # Enable/Disable filter

DTCFactor = 2           #The allowed values/ QUALITY FACTORS are 1(1/64),2(1/16) and 4(1/4 frequency data)

RGB = np.matrix([[0.299,     0.587,     0.114],
                 [-0.16864, -0.33107,   0.49970],
                 [0.499813, -0.418531, -0.081282]])

YCbCr = RGB.I

def applyDCT(frame):
    
    r,c = frame.shape
    Mr = np.concatenate([np.ones(DTCFactor), np.zeros(8-DTCFactor)])
    Mc = Mr
    # frame=np.reshape(frame[:,:,1],(-1,8), order='C')
    frame=np.reshape(frame,(-1,8), order='C')
    X=sft.dct(frame/255.0,axis=1,norm='ortho')
    #apply row filter to each row by matrix multiplication with Mr as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mr))
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 by using transposition .T:
    X=np.reshape(X.T,(-1,8), order='C')
    X=sft.dct(X,axis=1,norm='ortho')
    #apply column filter to each row by matrix multiplication with Mc as a diagonal matrix from the right:
    X=np.dot(X,np.diag(Mc))
    #shape it back to original shape:
    X=(np.reshape(X,(-1,r), order='C')).T
    #Set to zero the 7/8 highest spacial frequencies in each direction:
    #X=X*M    
    return X


def DCTFrame(frame):
    dctFrame = np.zeros(frame.shape)
    for i in range(frame.shape[2]):
        dctFrame[:,:,i] = applyDCT(frame[:,:,i])
    return dctFrame
    
def applyIDCT(frame):

    r,c= frame.shape
    X=np.reshape(frame,(-1,8), order='C')
    X=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    X=np.reshape(X,(-1,c), order='C')
    #Shape frame with columns of hight 8 (columns: order='F' convention):
    X=np.reshape(X.T,(-1,8), order='C')
    x=sft.idct(X,axis=1,norm='ortho')
    #shape it back to original shape:
    x=(np.reshape(x,(-1,r), order='C')).T

    return x

def IDCTFrame(frame):

    idctFrame = np.zeros(frame.shape)

    for i in range(frame.shape[2]):
        idctFrame[:,:,i] = applyIDCT(frame[:,:,i])

    return idctFrame

def app420(comp):
    """
    """
    r, c = comp.shape 
    #apply on chroma components with reduced size from Task 2
#     f = np.zeros((r*2,c*2))
    f = np.zeros((r,c))
    #apply on chroma components with reduced size from Task 2
    #f[0::2, 0::2] = comp
    f[0::2, 0::2] = comp[::2,::2]
    f[1::2, ] = f[0::2, ]
    f[:, 1::2] = f[:, 0::2]

    return f

def chomaSubSamp(y, cb, cr):
    """
    """
    r, c = y.shape
    # print frame.shape
    frame = np.zeros((r, c, 3))
    frame[:, :, 0] = y  #astype
    frame[:, :, 1] = app420(cb) #astype
    frame[:, :, 2] = app420(cr) #astype

    return frame

def filterFrame(frame, kernel):
    """
    """
    filtFrame = np.zeros(frame.shape)
    filtFrame[:, :, 0] = frame[:,:,0]
    filtFrame[:, :, 1] = sp.convolve2d(frame[:, :, 1], kernel, mode='same')
    filtFrame[:, :, 2] = sp.convolve2d(frame[:, :, 2], kernel, mode='same')

    return filtFrame

def test_function_removeZeros(x, factor):

    r, c = x.shape

    for k in range(factor, c + 1, factor):
        # x = np.delete(x, [k, k+1, k+2, k+3, k+4, k+5],  axis=1)
        x = np.delete(x, np.arange(k, k + (8-factor)), axis=1)


    for l in range(factor, r + 1, factor):
        # x = np.delete(x, [[l], [l+1], [l+2], [l+3], [l+4], [l+5]],  axis=0)
        x = np.delete(x, np.arange(l, l + (8-factor)).T, axis=0)

    # print x

    return x

def test_function_addZeros(x, factor):

    r, c = x.shape

    if factor == 4:
        increase = 2
    elif factor == 1:
        increase = 8
    else:
        increase = 4

    for j in range(factor, c * increase, 8):
        # x = np.insert(x, [j], [0, 0, 0, 0, 0, 0], axis=1)
        x = np.insert(x, [j], np.zeros(8-factor), axis=1)



    for i in range(factor, r * increase, 8):
        # x = np.insert(x, [i], [[0], [0], [0], [0], [0], [0]], axis=0)
        x = np.insert(x, [i], np.reshape(np.zeros(8-factor), (8-factor, 1)), axis=0)



    return x


def fillZeros(frame):

    r, c = frame.shape
    factor = r/60 #or c/80
    incRC = 480./r
    rem = 8-factor

    for i in range(factor, int(r*incRC), 8):
        frame = np.insert(frame, [i], np.zeros(rem).reshape(rem,1), axis=0)
    for i in range(factor, int(c*incRC), 8):
        frame = np.insert(frame,[i], np.zeros(rem), axis=1)

    return frame

def DCTWithZeros(frame, factor):
    """
    """
    f = np.zeros((480,640,3))

    for i in range(f.shape[2]):
        f[:,:,i] = test_function_addZeros(frame[:,:,i], factor)

    return f

def removeZeros(x):
    """
    """
    r,c = x.shape

    for k in range(2, c+1, 2):
        x = np.delete(x, np.arange(k, k+6),  axis=1)

    for l in range(2, r+1, 2):
        x = np.delete(x, np.arange(l, l+6).T, axis=0)

    return x

def DCTRemoveZeros(frame, f):

    rframe = np.zeros((60*f, 80*f, frame.shape[2]))

    for i in range(frame.shape[2]):
        rframe[:, :, i] = test_function_removeZeros(frame[:, :, i], f)

    return rframe#.astype('int8')

def upsample(frame, N):
    """
    """
    r, c = frame.shape
    # print frame.shape
    fu = np.zeros((r*N,c*N))
    fu[::N, ::N] = frame

    return fu

def Rgb2Ycbcr(frame):

    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(RGB, frame[i].T).T

    return xframe

def Ycbcr2Rgb(frame):

    xframe=np.zeros(frame.shape)

    for i in range(frame.shape[0]):
        xframe[i] = np.dot(YCbCr, frame[i].T).T#/255.

    return xframe


if __name__ == '__main__':

    import cv2
    x = cv2.imread('bb8.jpg')

    cv2.imshow('con', Rgb2Ycbcr(x)[:,:,0]/255.)
    cv2.waitKey(0)


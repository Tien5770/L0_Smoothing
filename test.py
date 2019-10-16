import numpy as np
import pylab
import PIL
import cv2
import math
from matplotlib import pyplot as plt

def circShift(mat,M):
    rows,cols = mat.shape
    if rows>=2 and cols>=2 and len(mat.shape)>=2:
        if int==type(M) and abs(M)<rows:
            #move up or down
            move = mat[:-M,:]
            remain = mat[-M:,:]
            new_mat = np.concatenate((remain,move),axis=0)
        elif (tuple==type(M) or list==type(M)) and abs(M[0])<rows and abs(M[1])<cols:
            move_y = mat[:-M[0],:]
            remain_y = mat[-M[0]:,:]
            after_y_ope = np.concatenate((remain_y,move_y),axis=0)

            move_x = after_y_ope[:,:-M[1]]
            remain_x = after_y_ope[:,-M[1]:]
            new_mat = np.concatenate((remain_x,move_x),axis=1)
        else:
            print("error in shape or operation")
            return None
    else:
        print("error in dimensions")
        return None

    return new_mat



def psf2otf(psf, outsize):
    if not (0 in psf):
        psf = np.double(psf)
        psf_size = np.shape(psf)
        psf_size = np.array(psf_size)
        pad_size = outsize - psf_size

        psf = np.lib.pad(psf,((0, pad_size[0]), (0, pad_size[1])),'constant')
        # Circularly shift otf so that the "center" of the PSF is at the (1,1) element of the array.
        psf = circShift(psf,(-math.floor(psf_size[0]/2),-math.floor(psf_size[1]/2)))
        # Compute the OTF
        otf = np.fft.fftn(psf,axes=(0,1))
        # Estimate the rough number of operations involved in the computation of the FFT.
        #computation of the FFT.
        n_elem = np.prod(psf_size,axis=0)
        n_ops = 0
        for i in range(0,np.ndim(psf)):
            nffts = n_elem / psf_size[i]
            n_ops = n_ops + psf_size[i] * np.log2(psf_size[i]) * nffts
        eps = 2.2204e-16
        mx1 = (abs(np.imag(otf[:])).max(0)).max(0)
        mx2 = (abs(otf[:]).max(0)).max(0)
        if (abs(np.imag(otf[:])).max(0)).max(0)/(abs(otf[:]).max(0)).max(0) <= eps * n_ops:
            otf = np.real(otf)
    else:
        otf = np.zeros(outsize)

    return otf



def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def L0_Smoothing(im,lam=2e-2,kappa=2.0):
    #S = im2double(im)
    S = im/255
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    N,M,D = im.shape
    sizeI2D = np.array([N,M])
    #fft
    otfFx = psf2otf(fx,sizeI2D)
    otfFy = psf2otf(fy,sizeI2D)
    Normin1 = np.fft.fft2(S,axes=(0,1))
    #F(∂x)∗F(∂x)+F(∂y)∗F(∂y)
    Denormin2 = pow(abs(otfFx),2) + pow(abs(otfFy),2)
    if D>1:
        tmp = np.zeros((N,M,D),dtype=np.double)
        for i in range(D):
            tmp[:,:,i] = Denormin2
        Denormin2 = tmp
    beta = 2 * lam
    betamax = 1e5
    while beta < betamax:
        # 1 + F(∂x)∗F(∂x)+F(∂y)∗F(∂y)
        Denormin = 1 + beta * Denormin2

        #h_v subproblem
        #x dimension
        shift_x = np.zeros((N,M,D))
        for i in range(0,D):
            shift_x[:, :, i] = circShift(S[:,:,i],[0,-1])
        h = shift_x - S

        #y dimension
        shift_y = np.zeros((N, M, D))
        for i in range(0, D):
            shift_y[:, :, i] = circShift(S[:, :, i], [-1, 0])
        v = shift_y - S

        if D==1:
            # (∂S/∂x)^2 + (∂S/∂y)^2
            val = (pow(h, 2) + pow(v, 2))
        else:
            val = np.sum((h ** 2 + v ** 2), 2)
        h[val < lam / beta] = 0
        v[val < lam / beta] = 0

        #S subproblem
        #求解二阶导数
        shift_h = np.zeros(h.shape)
        for i in range(0, h.shape[2]):
            shift_h[:, :, i] = circShift(h[:, :, i], [0, 1])
        hh = shift_h - h
        shift_v = np.zeros(v.shape)
        for i in range(0, v.shape[2]):
            shift_v[:, :, i] = circShift(v[:, :, i], [1, 0])
        vv = shift_v - v
        Normin2 = hh + vv
        FS = (Normin1 + beta * np.fft.fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(np.fft.ifft2(FS, axes=(0, 1)))
        beta *= kappa
        print('.')
    return S


def main():

    im = cv2.imread("images/pflower.jpg")
    S = L0_Smoothing(im, 0.01)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    S = S * 255
    image_out = "out/pflowerout.jpg"
    cv2.imwrite(image_out, S)
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.title("origin", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    im_out = cv2.imread("out/pflowerout.jpg")
    im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
    plt.imshow(im_out)
    plt.title("after", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.show()


main()




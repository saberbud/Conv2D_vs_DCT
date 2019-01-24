import numpy as np
from dct_funcs import *
from scipy import fftpack
np.random.seed(73)

def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')

coef_cos = prep_coef_cos(np.float32, Nd=8)
print("coef_cos.shape: " + str(coef_cos.shape))
print("coef_cos: \n" + str(coef_cos) + '\n')

im_c = np.random.randint(0,255,[8,8])

'''
n = 8
Gout = np.zeros([n,n])
DT = np.zeros([n,n])
for iu in range(n):
    for iv in range(n):
        #Gout[iu,iv] = dct_loop_1(image, coef_cos, iu, iv)
        g, dt = dct_loop_1(im_c, coef_cos, iu, iv, np.float32, np.float32, np.float32)
        Gout[iu,iv] = g
        DT[iu,iv] = dt
'''

#type1=np.float32; type2=np.float32; type3=np.float32
type1=np.float64; type2=np.float64; type3=np.float64

Gout, DT = DCT(im_c, coef_cos, type1, type2, type3)

print("Gout.shape: " + str(Gout.shape))
print("Gout: \n" + str(Gout))

print('\nDT: \n' + str(DT))
print('\nSum DT= ' + str(np.sum(DT)))

nrun = 10000
DTS = np.zeros([nrun])
for j in range(nrun):
    if(j%500==0):print('Running ' + str(j))
    im_c = np.random.randint(0,255,[8,8])
    Gout, DT = DCT(im_c, coef_cos, type1, type2, type3)
    DTS[j] = np.sum(DT)

print('DTS sum= ' + str(np.sum(DTS)))
print('DTS mean= ' + str(np.mean(DTS)))
print('DTS std= ' + str(np.std(DTS)))


'''
Dout = dct_2d(im_c)
print("\nDout.shape: " + str(Dout.shape))
print("Dout: \n" + str(Dout))
'''














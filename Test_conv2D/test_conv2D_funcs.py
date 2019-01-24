import numpy as np
import h5py
from conv2d_funcs import *
np.random.seed(73)


f_code=h5py.File('./code_884_1x16_mix_4bit.h5','r')
#f_code=h5py.File('./code_16164_1x64_mix_4bit.h5','r')
#f_code=h5py.File('./code_32324_1x256_mix_4bit.h5','r')
code=f_code['data'][:]
print("code shape: " + str(code.shape))
print('code type: ' + str(code.dtype) + '\n')
for ic in range(code.shape[3]):
    print(str(ic) + " 2D part of code: \n" + str(code[:,:,0,ic]))
    print('')

nx = code.shape[0]
ny = code.shape[1]
NC = 1

#Use some fake data
data_in = np.random.randint(0, 255, [1,nx,ny,NC])

print('NP conv')
CO={}; DT={}
for i in range(4):
    CO[i],DT[i] = conv2d_ele_np(data_in[0,:,:,0], code[:,:,0,i], np.uint8, np.int8, np.int32)
    print("CO " + str(i) + ": " + str(CO[i]) + ' ; DT= ' + str(DT[i]))


print('\nLoop conv')
#type1=np.uint8; type2=np.int8; type3=np.int32
#type1=np.uint8; type2=np.float32; type3=np.float32
#type1=np.int32; type2=np.int32; type3=np.int32
#type1=np.float32; type2=np.float32; type3=np.float32
type1=np.float64; type2=np.float64; type3=np.float64

CO={} 
DT=np.zeros([4])
for i in range(4):
    co,dt = conv2d_ele_loop(data_in[0,:,:,0], code[:,:,0,i], type1, type2, type3)
    CO[i] =  co
    DT[i] = dt
    print("CO " + str(i) + ": " + str(CO[i]) + ' ; DT= ' + str(DT[i]))

print('\nSum time= ' + str(np.sum(DT)))

nrun = 10000
DTS = np.zeros([nrun])
for j in range(nrun):
    if(j%2000==0):print('Running ' + str(j))
    dts = 0.
    data_in = np.random.randint(0, 255, [1,nx,ny,NC])
    for i in range(4):
        co,dt = conv2d_ele_loop(data_in[0,:,:,0], code[:,:,0,i], type1, type2, type3)
        dts = dts + dt

    DTS[j] = dts

print('DTS sum= ' + str(np.sum(DTS)))
print('DTS mean= ' + str(np.mean(DTS)))
print('DTS std= ' + str(np.std(DTS)))









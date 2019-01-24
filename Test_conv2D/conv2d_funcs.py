import numpy as np
import time

def conv2d_ele_np(data_in, kele, type_d, type_k, type_o):
    assert(data_in.shape == kele.shape)
    data_in  = data_in.astype(type_d)
    kele = kele.astype(type_k)
    
    st=time.clock()
    out = np.sum(data_in*kele)
    et=time.clock()

    dt = et - st
    out.astype(type_o)

    return out, dt


def conv2d_ele_loop(data_in, kele, type_d, type_k, type_o):
    assert(data_in.shape == kele.shape)
    data_in = data_in.astype(type_d)
    kele = kele.astype(type_k)

    out = np.array([0])
    out = out.astype(type_o)

    st=time.clock()

    for ix in range(data_in.shape[0]):
        for iy in range(data_in.shape[1]):
            out[0] = out[0] + data_in[ix,iy] * kele[ix,iy]

    et=time.clock()

    dt = et - st
    out = out.astype(type_o)

    return out[0], dt

















import numpy as np
import time

def prep_coef_cos(type_cc, Nd=8):
    coef_cos = np.zeros([Nd,Nd])
    coef_cos.astype(type_cc)
    for i in range(Nd):
        for j in range(Nd):
            coef_cos[i,j] = np.cos(np.pi*i*(2*j+1)/(2*Nd))

    coef_cos.astype(type_cc)

    return coef_cos

def dct_loop_1(image_in, coef, u, v, type_d, type_k, type_o, N=8):
    assert(image_in.shape==coef.shape)
    image_in = image_in.astype(type_d)
    coef = coef.astype(type_k)

    a_u = 1./np.sqrt(N*4)
    if u>0:
        a_u=1./np.sqrt(N*2)
    
    a_v = 1./np.sqrt(N*4)
    if v>0:
        a_v=1./np.sqrt(N*2)

    #a_v = a_v*4. #remove 4. in loop
    a_v = a_v*a_u*4. #remove 4. and a_u in loop
    
    aau=np.array([a_u])
    aau=aau.astype(type_k)
    a_u = aau[0]

    aav=np.array([a_v])
    aav=aav.astype(type_k)
    a_v = aav[0]
    
    
    G=np.array([0])
    G=G.astype(type_o)

    st=time.clock()

    for i in range(N):
        for j in range(N):            
            #G[0] = G[0] + 4.*a_u*a_v*image_in[i,j]*coef[u,i]*coef[v,j]
            #G[0] = G[0] + a_u*a_v*image_in[i,j]*coef[u,i]*coef[v,j]
            G[0] = G[0] + a_v*image_in[i,j]*coef[u,i]*coef[v,j]


    et=time.clock()
    G=G.astype(type_o)


    dt = et - st

    return G[0], dt


def DCT(image_in, coef, type_d, type_k, type_o, N=8):
    Gout = np.zeros([N,N])
    Gout = Gout.astype(type_o)
    DT = np.zeros([N,N])
    for iu in range(N):
        for iv in range(N):
            g, dt = dct_loop_1(image_in, coef, iu, iv, type_d, type_k, type_o)
            Gout[iu,iv] = g
            DT[iu,iv] = dt

    Gout = Gout.astype(type_o)

    return Gout, DT

















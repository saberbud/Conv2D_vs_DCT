import numpy as np
import tensorflow as tf
import h5py
np.random.seed(73)

###############set gpu###############
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
#####################################

input_image_size = 8
NC = 1
print("input_image_size: " + str(input_image_size))

#Can change different code files to see effect
f_code=h5py.File('./code_884_1x16_mix_4bit.h5','r')
code=f_code['data'][:]
code=code.astype(np.float32)
print("code shape: " + str(code.shape))
print("code: " + str(code))
print("oth 2D part of code: " + str(code[:,:,0,0]))

stride_size = code.shape[0]

#Conv2D kernel
compW = tf.constant(code)
print("\ncompW: " + str(compW))

## input layer
with tf.name_scope('DATA'):
    Data_train = tf.placeholder(tf.float32,shape=[None,input_image_size,input_image_size,NC])
    print(Data_train)


with tf.name_scope('Compress_Conv'):
    comp = tf.nn.conv2d(Data_train, compW, strides=[1, stride_size, stride_size, 1], padding='VALID')
    print(comp)

#Use some fake data
data_in = np.random.randint(0, 255, [1,input_image_size,input_image_size,NC])

#Run example compression:
with tf.Session(config=config) as sess:
    comp_o=sess.run(comp,feed_dict={Data_train:data_in})

#Check output:
print("\ncomp_o shape: " + str(comp_o.shape))
print("\ncomp_o: \n" + str(comp_o))

#co_0 = np.sum(code[:,:,0,0]*data_in[0,:,:,0])
#print("\nco_0= " + str(co_0))
co_np={}
for i in range(4):
    co_np[i]=np.sum(code[:,:,0,i]*data_in[0,:,:,0])
    print("co_np " + str(i) + ": " + str(co_np[i]))













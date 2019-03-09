import sys
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time


nets_path = r'slim'
if nets_path not in sys.path:
   sys.path.insert(0,nets_path)
else:
   print('already add slim')
import tensorflow as tf                                   #引入头文件
from PIL import Image
from matplotlib import pyplot as plt
from nets.nasnet import pnasnet
import numpy as np
from datasets import imagenet
from tensorflow.python.framework import graph_util 


path='/home/yang/models-master/research/slim/cancer2/'

#将所有的图片resize成100*100
w=32
h=32
c=3


#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.png'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]



slim = tf.contrib.slim

tf.reset_default_graph() 
arg_scope = pnasnet.pnasnet_large_arg_scope()                  #获得模型命名空间

image_size = pnasnet.build_pnasnet_large.default_image_size       #获得图片输入尺寸
#labels = imagenet.create_readable_names_for_imagenet_labels()     #获得数据集标签

image_size=32
input_imgs = tf.placeholder(tf.float32, [None, image_size,image_size,3]) #定义占位符
#占位符

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


x1 = 2 *( input_imgs / 255.0)-1.0                                 #归一化图片

with slim.arg_scope(arg_scope):

    logits, end_points = pnasnet.build_pnasnet_large(x1,num_classes = 1001, is_training=False)   
    prob = end_points['Predictions']
    y = tf.argmax(prob,axis = 1)                                  #获得结果的输出节点



  



checkpoint_file = '/home/yang/pnasnet-5_large_2017_12_13/model.ckpt'       #定义模型路径
saver = tf.train.Saver()                                                #定义saver，用于加载模型
n_epoch=500

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

with tf.Session() as sess:                                              #建立会话
    saver.restore(sess, checkpoint_file)                            #载入模型
    print("hello")

   

    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
   # for tensor_name in tensor_name_list:
         #     print(tensor_name,'\n')

    feature = tf.get_default_graph().get_operation_by_name("final_layer/predictions").outputs[0]

    #feature = tf.get_default_graph().get_operation_by_name("final_layer/FC/BiasAdd").outputs[0]
    print(feature)

    logits2= tf.layers.dense(inputs=feature, 
                        units=2, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.nn.l2_loss)
    print(logits2)
    loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits2)
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits2,1),tf.int32), y_)    
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #训练和测试数据，可将n_epoch设置更大一些

    n_epoch=1000
    batch_size=64
#sess=tf.InteractiveSession()  
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
         start_time = time.time()
       #  print(""+ str(n_batch)
    
    #training
         train_loss, train_acc, n_batch = 0, 0, 0
         for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
              _,err,ac=sess.run([train_op,loss,acc], feed_dict={input_imgs: x_train_a, y_: y_train_a})
             # print(err)
             # print("err"+acc)
              train_loss += err; train_acc += ac; n_batch += 1
         print("   train loss: %f" % train_loss/ n_batch)
         print("   train acc: %f" % train_acc/ n_batch)
    
    #validation
         val_loss, val_acc, n_batch = 0, 0, 0
         for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
              err, ac = sess.run([loss,acc], feed_dict={input_imgs: x_val_a, y_: y_val_a})
              val_loss += err; val_acc += ac; n_batch += 1
         print("   validation loss: %f" % val_loss/ n_batch)
         print("   validation acc: %f" % val_acc/ n_batch)

sess.close()


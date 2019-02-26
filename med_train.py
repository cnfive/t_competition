# coding=utf-8
import csv
import datetime
import time
import tensorflow as tf
import re
from numpy import matrix,mat
import numpy as np
from numpy import shape,array
import math


import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


csv_file=csv.reader(open("jinnan_round1_train_20181.csv","r"))

sample=[]
sample_y=[]
c=0
for line in csv_file:
   c=c+1
   if c>1:
     print(line)
     print("#"*20)

     list_line=[]
     for key in line:
       print(key)
       if key.find("sample")==-1:
         if key=='':
            key="0"
         
         #把时间转换成分钟 计量
         p1 = r"(\d{4}/\d{1,2}/\d{1,2})" #
         pattern1 = re.compile(p1)#
         matcher1 = re.search(pattern1,key)#在源文本中搜索符合正则表达式的部分
         if matcher1:
            #print(matcher1)
           
            key=key.replace(matcher1.group(0), "");

         if key.find(":")>-1 and key.find("-")==-1:

             time=key.split(":")
             hour=0
             if time[0]!='':
                  hour=int(time[0])

             minite=int(time[1])
             print(hour*60+minite)

             list_a=(hour*60+minite)

         if key.find("-")>-1:
             key=key.replace('"',':')
             key=key.replace('；',':')
             key=key.replace(';',':')
             key=key.replace('分','')
             key=key.replace('::',':')
             t=key.split("-")
             #print(t)
         
             time_count=[]
             i=0
             for time_string in t:

                 if time_string.find(':')==-1:

                         str_list=list(time_string)

                         str_list.insert(-3,':')

                         time_string="".join(str_list)
                        
                 time=time_string.split(":")
                 hour=0
                 if time[0]!='':
                      hour=int(time[0])
                 if time[1]=='':
                      time[1]=00
                 minite=int(time[1])
                # print(hour*60+minite)
                 time_count.append(hour*60+minite)
                 #i=i+1
             time2=time_count[0]+time_count[1]
             print(time2)
             list_a=time2

         if key.find(":")==-1 and key.find("-")==-1:
             list_a=float(key)

         list_line.append(list_a)
     sample_y.append(list_line[-1])
     list_line.pop()
     sample.append(list_line)
     #sample_y.append(list_line[-1])
print(sample)
mat1=matrix(sample)
print(mat1.shape)
max_index=mat1.argmax(axis=0)
print(max_index)
min_index=mat1.argmin(axis=0)
print(min_index)

for m in min_index:
      print(m)
           
a = np.random.randint(50,size= (4,5))
print (a)
print (a.argmin(axis=0) )

max_value=np.max(mat1,0)
print(max_value)
min_value=np.min(mat1,0)

print(min_value)
x_in=(mat1-min_value)/(max_value-min_value)
print(x_in)

#按行输出矩阵
shapes=shape(x_in)
print(shapes)
print(shapes[0])


print(sample_y)


#对sample_y 分类词典
dic={}
num=0
d = {}
L=d.fromkeys(sample_y)

print(L)

for y in L:
   dic[y]=num
   num=num+1
print(dic)

ylabel=[]
for y in sample_y:
      ylabel.append(int(dic[y]))


print(ylabel)
yl=matrix(ylabel)







learning_rate=0.01

global_step = tf.Variable(-1, trainable=False, name='global_step')


num_layer=2

num_units=128

category_num=73

y_label=tf.placeholder(dtype=tf.float32, shape=[None,1],name='y_label')



x=tf.placeholder(dtype=tf.float32, shape=[None,42,1],name='x')


 # Variables
keep_prob = tf.placeholder(tf.float32, [])



def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def lstm_cell(num_units, keep_prob=0.5):
    cell = tf.nn.rnn_cell.LSTMCell(num_units, reuse=tf.AUTO_REUSE)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)


keep_prob=0.5

cell_fw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
cell_bw = [lstm_cell(num_units, keep_prob) for _ in range(num_layer)]
   
inputs2 = tf.unstack(x, 42, axis=1)
#print(len(inputs))
#for i in inputs:
    #print(i)
#inputs=x
#print(inputs2)
output, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs2, dtype=tf.float32)

print("output shape:",shape(output))

#input()
# Output Layer
#output = tf.reshape(output, [42, -1])
with tf.variable_scope('outputs'):
     w = weight([num_units * 2, category_num])
     b = bias([category_num])
     pred = tf.matmul(output[-1], w) + b
     print("w:",w)
     print("b:",b)
     print("y.shape:",pred.shape)

        
     #y_predict = tf.cast(tf.argmax(y2, axis=1), tf.int32)
     #print('Output Y shape:', y_predict.shape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
value=tf.argmax(pred,1)
value2=y_label

print(value2)

correct_pred = tf.equal(tf.argmax(pred,1), tf.cast(y_label,tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess=tf.Session() 
sess.run(init)
  
saver = tf.train.Saver(max_to_keep=2)
 
for j in range(1000):

  print("$"*100)
  print("j=",j)
  input_value=[]
  y2=[]
  print("shapes:",int(shapes[0]/150))
  acc=0

  #for i in range(1):
  for i in range(math.ceil(shapes[0]/150)):
    if i<9:
    
       input_value=x_in[i*150:(i+1)*150,:]
    
    # y=1000*y
       y2=yl[0,i*150:(i+1)*150]
    else:
       y2=yl[0,-151:-1]
      # print(y2)

       input_value=x_in[-151:-1,:]

     #y=y.eval(session=sess)
     #print("y2:",y2)

    y3=matrix(y2)
    # print("input_value shape:",input_value)
     #input_value=tf.reshape(input_value,[150,42])
    input_x=sess.run(tf.expand_dims(input_value,2))
    # print(input_x)
    y4=sess.run(tf.reshape(y3,[150,1]))
    predict,acc,v,v2=sess.run([value,accuracy,value,value2], feed_dict={x: input_x,y_label:y4})
     #value=sess.run(c)
    #print(sess.run(tf.argmax(c,1)))
    print("predict:",predict)
    print("true value:",y3)
    print(acc)
    #print(v)
    #print(v2)
     #print("loss:",loss)
  if j>100 :
       saver.save(sess, "Model/model.ckpt",global_step=j)

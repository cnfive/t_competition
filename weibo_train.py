import tensorflow as tf
import os
from numpy import shape,array,matrix

from bert_serving.client import BertClient
bc = BertClient()
filename="./weibo_train_data.txt"

file_txt=open(filename,"+r")

user={}
txt=[]
for line in file_txt:

      print(line)
      s=line.split("	")
      print(s[0])
      print(s[6])
      user[s[0]]=""
      txt.append(s)

print(user)
num=1

for key in user:
     user[key]=num
     num=num+1

print(user)
print(len(txt))


     #print(em_list)


seq_length2=1300

def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length2, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        print(shape(one_hot_positions))
        print(shape(logits))
        print(shape(log_probs))
        predict=tf.argmax(one_hot_positions * log_probs)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss,predict,logits



#start_positions = features["start_positions"]
#end_positions = features["end_positions"]

#start_loss = compute_loss(start_logits, start_positions)
#end_loss = compute_loss(end_logits, end_positions)
      #计算答案前后位置的损失函数

#x=tf.placeholder(dtype=tf.float32, shape=[None,42,1],name='x')

final_hidden_matrix=tf.placeholder(dtype=tf.float32, shape=[None,768],name='final_hidden_matrix')
start_positions=tf.placeholder(dtype=tf.int32, shape=[None,1],name='start_positions')
end_positions=tf.placeholder(dtype=tf.int32, shape=[None,1],name='end_positions')
middle_positions=tf.placeholder(dtype=tf.int32, shape=[None,1],name='middle_positions')
batch_size =128        #批次数量大小
seq_length =1          #序列长度
hidden_size =768       #隐藏单元大小

  #初始化权重矩阵
output_weights = tf.get_variable(
      "output_weights", [3*1300, hidden_size],                      #1300是统计出来语料赞，评，转，个数做为分类，最大不超过1300个分类
      initializer=tf.truncated_normal_initializer(stddev=0.02))
  #初始化偏置项
output_bias = tf.get_variable(
      "output_bias", [3*1300], initializer=tf.zeros_initializer())

#final_hidden_matrix = tf.reshape(final_hidden,
 #                                  [batch_size * seq_length, hidden_size])


  #回归训练，求解y=ax+b  ，这种模型结构，a=output_weights，b=output_bias，y=logits
logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)   #矩阵相乘
logits = tf.nn.bias_add(logits, output_bias)                                #加上偏置项

#logits = tf.reshape(logits, [batch_size, seq_length, 2])  #把输出转换成[每批数量，序列长度，2] 这种格式[2维数组数,行数，列数]
#logits = tf.transpose(logits, [2, 0, 1])                  #对上一步的结果转置，[2,0,1]代表[列数，2维数组数，行数]

unstacked_logits = tf.unstack(logits, axis=1)             #对矩阵在行上拆分

#(start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])
start_logits,middle_logits, end_logits = tf.reshape(unstacked_logits[0:1300],[128,1300]), tf.reshape(unstacked_logits[1300:2600],[128,1300]) ,tf.reshape(unstacked_logits[2600:3900],[128,1300])
#加一层全连接

start_loss,start_predict,probs = compute_loss(start_logits, start_positions)
end_loss,end_predict,_ = compute_loss(end_logits, end_positions)
middle_loss,middle_predict,_ = compute_loss(middle_logits, middle_positions)

#return (start_logits, end_logits)                         #返回起始位置，结束位置
total_loss = (start_loss + end_loss+middle_loss) / 3.0
learning_rate=0.01
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

correct_pred = tf.equal(tf.argmax(start_predict,1), tf.cast(start_positions,tf.int64))
pred=tf.argmax(start_predict,1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess=tf.Session() 
sess.run(init)


  
saver = tf.train.Saver(max_to_keep=2)


s_d={}
m_d={}
e_d={}

for i in range(int(len(txt)/128)):
     list_s=txt[i*128:(i+1)*128]
     print(list_s)
    

     for s in  list_s:
         #print(s[6])

         if s[3] not in s_d:
            s_d[s[3]]=1
         else:
            s_d[s[3]]+=1

         if s[4] not in m_d:
            m_d[s[4]]=1
         else:
            m_d[s[4]]+=1

         if s[5] not in e_d:
            e_d[s[5]]=1
         else:
            e_d[s[5]]+=1

 

print("s_d=",len(s_d))
print("m_d=",len(m_d))     
print("e_d=",len(e_d))

#s_d= 1243
#m_d= 527
#e_d= 1020
s_dic={}
m_dic={}
e_dic={}

i=0

#把赞，评，转，数转换成分类词典
for k in s_d:
    s_dic[k]=i
    i=i+1
i=0
for k in m_d:
    m_dic[k]=i
    i=i+1   
i=0
for k in e_d:
    e_dic[k]=i
    i=i+1 


#for i in range(int(len(txt)/128)):
for j in range(1000):                 #迭代1000轮
  #for i in range(10):
  for i in range(int(len(txt)/128)):   #每次读取128条
     list_s=txt[i*128:(i+1)*128]
     #print(list_s)
     batch_list=[]
     start=[]
     end=[]
     middle=[]

     for s in  list_s:
         #print(s[6])

         batch_list.append(s[6])
         start.append(s_dic[s[3]] )
         middle.append(m_dic[s[4]])
         end.append(e_dic[s[5]])
     #print(batch_list)
     em_list=bc.encode(batch_list)
     start=sess.run(tf.reshape(start,[128,1]))
     end=sess.run(tf.reshape(end,[128,1]))
     middle=sess.run(tf.reshape(middle,[128,1]))
     print(shape(start))

     _,loss=sess.run([optimizer,total_loss], feed_dict={final_hidden_matrix:em_list,start_positions:start,middle_positions:middle,end_positions:end })
     #loss=sess.run([start_logits], feed_dict={final_hidden_matrix:em_list })
     #print(shape(p))
     print(loss)
     #print(p)

   

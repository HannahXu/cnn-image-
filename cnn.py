import os
from PIL import Image
import numpy as np
import tensorflow as tf

data_dir=r'C:\Users\徐涵\Desktop\gan-cifar\data'
train=False
model_path=r"H:\model\image_model"#注意地址的确不能有中文

def read_data(data_dir):
    datas=[]
    labels=[]
    fpaths=[]#????]

    for fname in os.listdir(data_dir):#抽单个图片出来
        fpath=os.path.join(data_dir,fname)#单个图片的路径
        fpaths.append(fpath)
        image=Image.open(fpath)
        data=np.array(image)/255.0
        label=int(fname.split("_")[0])#_符号前面的数,0??????
        datas.append(data)
        labels.append(label)
    datas=np.array(datas)
    labels=np.array(labels)

    print('shape of datas:{}\tshape of labels:{}'.format(datas.shape,labels.shape))
    return fpaths,datas,labels
fpaths,datas,labels=read_data(data_dir)

num_classes=len(set(labels))#set创建集合，作用？？？？

datas_placeholder=tf.placeholder(tf.float32,[None,32,32,3])
labels_placeholder=tf.placeholder(tf.int32,[None])

dropout_placeholder=tf.placeholder(tf.float32)

conv0=tf.layers.conv2d(datas_placeholder,20,5,activation=tf.nn.relu)#调参
pool0=tf.layers.max_pooling2d(conv0,[2,2],[2,2])

conv1=tf.layers.conv2d(pool0,40,4,activation=tf.nn.relu)#调参
pool1=tf.layers.max_pooling2d(conv1,[2,2],[2,2])

flatten=tf.contrib.layers.flatten(pool1)

fc=tf.layers.dense(flatten,400,activation=tf.nn.relu)#400哪来的？？？？

dropout_fc=tf.layers.dropout(fc,dropout_placeholder)

logits=tf.layers.dense(dropout_fc,num_classes)

predicted_labels=tf.arg_max(logits,1)#参数是啥？？？？debug

losses=tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder,num_classes),#one_hot???????
    logits=logits
)
mean_loss=tf.reduce_mean(losses)

optimizer=tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)

saver=tf.train.Saver()

with tf.Session() as sess:
    if train:
        print("训练模式")
        sess.run(tf.global_variables_initializer())
        train_feed_dict={
            datas_placeholder:datas,
            labels_placeholder:labels,
            dropout_placeholder:0.25#啥含义？？？？
        }
        for step in range(1500):#为啥没出现batch
            _,mean_loss_val=sess.run([optimizer,mean_loss],feed_dict=train_feed_dict)#运行所有能得到？
            if step%10 ==0:
                print('step={}\tmean loss={}'.format(step,mean_loss_val))
        saver.save(sess,model_path)
        print("训练结束，保存模型到{}".format(model_path))
    else:
        print("测试模型")
        saver.restore(sess,model_path)
        print("从{}载入模型".format(model_path))
        label_name_dict={
            0:'飞机',
            1:'汽车',
            2:'鸟'

        }
        test_feed_dict={
            datas_placeholder:datas,#是在这改测试入口？？？？
            labels_placeholder:labels,
            dropout_placeholder:0
        }
        predicted_labels_vals=sess.run(predicted_labels,feed_dict=test_feed_dict)
        for fpath ,real_label,predicted_label in zip(fpaths,labels,predicted_labels_vals):
            #为啥有zip:这样一句就可以同时遍历3个变量
            real_label_name=label_name_dict[real_label]
            predicted_label_name=label_name_dict[predicted_label]

            print('{}\t{}=>{}'.format(fpath,real_label_name,predicted_label_name))



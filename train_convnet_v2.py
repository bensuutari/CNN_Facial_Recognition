import tensorflow as tf
import os
from os import listdir
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
import csv

imsize=56#define how big the square image should be along a side, e.g. imsize=100 means that images will be converted to 100x100 images, this value should be divisible by 4
num_epochs=200


########################################################################################################################################
#Obtain all images from the data directory, Yale Face Database obtained from http://vision.ucsd.edu/content/yale-face-database
datadirectory=os.getcwd()+'/data/'
allimgnames=os.listdir(datadirectory)
allimgnames.remove('Readme.txt')#remove the Readme file so we don't include it in our analysis

allimgnamesdict=dict()
for i in allimgnames:
    subjnum=int(i[7:9])
    if subjnum in allimgnamesdict.keys():
        allimgnamesdict[subjnum].append(i)
    else:
        allimgnamesdict[subjnum]=list()
        allimgnamesdict[subjnum].append(i)
n_classes=len(allimgnamesdict)
########################################################################################################################################

train_set_names=list()
train_set_labels=list()
test_set_names=list()
test_set_labels=list()

for i in  allimgnamesdict:
    random.shuffle(allimgnamesdict[i])
    hm_test=int(0.2*len(allimgnamesdict[i]))#how many examples to take for the test set, here we use 20% of data for test and 80% for training
    test_set_names=test_set_names+allimgnamesdict[i][0:hm_test]
    test_set_labels=test_set_labels+[int(i)]*hm_test
    train_set_names=train_set_names+allimgnamesdict[i][hm_test:]
    train_set_labels=train_set_labels+[int(i)]*(len(allimgnamesdict[i])-hm_test)

##################################################
#shuffle the training and testing sets 
combinetrain=zip(train_set_names,train_set_labels)
combinetest=zip(test_set_names,test_set_labels)

random.shuffle(combinetrain)
random.shuffle(combinetest)

train_set_names[:],train_set_labels[:]=zip(*combinetrain)
test_set_names[:],test_set_labels[:]=zip(*combinetest)
##################################################

'''
print allimgtensor
print type(allimgtensor)
print allimglabels
print type(allimaglabelstensor)
'''

testarray=np.zeros((len(test_set_names),imsize*imsize))
testlabels=np.zeros((len(test_set_names),n_classes))
counter=0
for i in test_set_names:
    testlabels[counter,int(test_set_names[counter][7:9])-1]=1#labels as one hot
    im=Image.open(datadirectory+i)
    im=im.resize((imsize,imsize),Image.NEAREST)
    im=np.asarray(im)/255.0
    testarray[counter,:]=im.reshape((imsize*imsize,))
    counter+=1
    '''
    plt.imshow(im)
    plt.show()
    '''

trainarray=np.zeros((len(train_set_names),imsize*imsize))
trainlabels=np.zeros((len(train_set_labels),n_classes))
counter=0
for i in train_set_names:
    trainlabels[counter,int(train_set_names[counter][7:9])-1]=1#labels as one hot
    im=Image.open(datadirectory+i)
    im=im.resize((imsize,imsize),Image.NEAREST)
    im=np.asarray(im)/255.0
    trainarray[counter,:]=im.reshape((imsize*imsize,))
    counter+=1

########################################################################################################################################
x = tf.placeholder('float', [None, imsize*imsize])
y = tf.placeholder('float',[None,n_classes])

keep_rate = 0.8#0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([(imsize/4)*(imsize/4)*64,1024])),#imsize/4 because each max pool has a stride of 2, so it reduces dimensionality by 2 twice or 1/2^2
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, imsize, imsize, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, (imsize/4)*(imsize/4)*64])#imsize/4 because each max pool has a stride of 2, so it reduces dimensionality by 2 twice or 1/2^2
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    print 'prediction shape='
    print(prediction.get_shape())
    print 'logits shape='
    print(y.get_shape())
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        
        
        acclist=list()
        losslist=list()
        for epoch in range(num_epochs):
            epoch_loss = 0
            _, c = sess.run([optimizer, cost], feed_dict={x: trainarray, y: trainlabels})
            epoch_loss += c
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            acclist.append(accuracy.eval({x:testarray, y:testlabels}))
            losslist.append(epoch_loss)
            print('Epoch '+ str(epoch+1) + '/'+str(num_epochs)+','+' loss:'+str(epoch_loss),'Test Accuracy:',accuracy.eval({x:testarray, y:testlabels}))
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:testarray, y:testlabels}))
        return acclist,losslist

acclist,losslist=train_neural_network(x)

with open(os.getcwd()+'/results/accuracy', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(acclist)

with open(os.getcwd()+'/results/loss', 'wb') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(losslist)
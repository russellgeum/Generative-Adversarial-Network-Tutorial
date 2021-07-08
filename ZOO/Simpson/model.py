import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.backend.tensorflow_backend import set_session

get_ipython().system('pip install tqdm')

tf.__version__

SimpsonFileName = "/content/drive/My Drive/Colab Notebooks/Datasets/Simpson"
Local_SimpsonFileName = "C:/Users/Maritimus/Desktop/Simpson DCGAN/Simpson"

# 이미지파일 경로를 저장할 리스트
DirectoryList = []

for dirname, _, filenames in os.walk(Local_SimpsonFileName):
    for filename in filenames:
        DirectoryList.append(os.path.join(dirname, filename))

print (DirectoryList)
np.random.shuffle(DirectoryList)

print (np.shape(DirectoryList))

length = int(len(DirectoryList))

# 경로의 이미지를 불러오고 Array로 변환한 다음의 이미지를 추가할 리스트
SimpsonImage = []

for number in tqdm(range (length)):
    
    Image = load_img(DirectoryList[number], target_size = (128, 128))
    Image = img_to_array(Image)
    SimpsonImage.append(Image/255.)

print (np.shape(SimpsonImage))

# 데이터 셔플, Generator 함수, Discriminator 함수를 정의
def Build_Shuffle_BatchData (BatchSize, Input) :
    
    np.random.shuffle (Input)
    TempGetImage = Input[0 : BatchSize]
    Output = TempGetImage
    
    return Output


def Build_Generator (inputs): 
    with tf.variable_scope("Generator_Variable"):
        
        output = tf.layers.dense(inputs, 8*8*1024)
        output = tf.reshape(output, [-1, 8, 8, 1024])
        output = tf.nn.leaky_relu (output)
        print ("Generator Size")
        print (np.shape(output))
    
        output = tf.layers.conv2d_transpose(output, 512, Kernel_Size, strides = (2, 2), 
                                            padding = "SAME", use_bias = False)
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        print (np.shape(output))
        
        output = tf.layers.conv2d_transpose(output, 256, Kernel_Size, strides = (2, 2), 
                                            padding = "SAME", use_bias = False)
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        print (np.shape(output))
        
        output = tf.layers.conv2d_transpose(output, 128, Kernel_Size, strides = (2, 2), 
                                            padding = "SAME", use_bias = False)
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        print (np.shape(output))
        
        output = tf.layers.conv2d_transpose(output, 64, Kernel_Size, strides = (2, 2), 
                                            padding = "SAME", use_bias = False)
        output = tf.layers.batch_normalization(output, training = IsTraining)
        output = tf.nn.leaky_relu (output)
        print (np.shape(output))
        
        output = tf.layers.conv2d_transpose(output, 3, Kernel_Size, strides = (1, 1), 
                                            padding = "SAME")
        print (np.shape(output))
        output = tf.tanh (output)
        
    return output
 
    
def Build_Discriminator (inputs, reuse = None):
    with tf.variable_scope("Discriminator_Variable") as scope:
        
        if reuse:
            scope.reuse_variables()

        outputs = tf.layers.conv2d(inputs, 64, Kernel_Size, strides = (2, 2), 
                                  padding = "SAME", use_bias = True)
        outputs = tf.nn.leaky_relu (outputs)
        print ("Disctriminator Size")
        print (np.shape(outputs))
        # 64 64 64
        
        outputs = tf.layers.conv2d(outputs, 128, Kernel_Size, strides = (2, 2), 
                                  padding = "SAME", use_bias = False)
        outputs = tf.layers.batch_normalization(outputs, training = IsTraining)
        outputs = tf.nn.leaky_relu (outputs)
        print (np.shape(outputs))
        # 32 32 128
        
        outputs = tf.layers.conv2d(outputs, 256, Kernel_Size, strides = (2, 2), 
                                  padding = "SAME", use_bias = False)
        outputs = tf.layers.batch_normalization(outputs, training = IsTraining)
        outputs = tf.nn.leaky_relu (outputs)
        print (np.shape(outputs))
        # 16 16 256
        
        outputs = tf.layers.conv2d(outputs, 512, Kernel_Size, strides = (1, 1), 
                                  padding = "SAME", use_bias = False)
        outputs = tf.layers.batch_normalization(outputs, training = IsTraining)
        outputs = tf.nn.leaky_relu (outputs)
        print (np.shape(outputs))
        # 16 16 512

        outputs = tf.layers.conv2d(outputs, 1024, Kernel_Size, strides = (2, 2), 
                                  padding = "SAME", use_bias = False)
        outputs = tf.layers.batch_normalization(outputs, training = IsTraining)
        outputs = tf.nn.leaky_relu (outputs)
        print (np.shape(outputs))
        
        outputs = tf.layers.flatten(outputs)
        outputs = tf.layers.dense(outputs, 1, activation = None)
        
    return outputs

def Build_GetNoise (batch_size, noise_size):
    return np.random.uniform(-1., 1., size=[batch_size, noise_size])


TotalEpoch = 600
BatchSize = 100
NoiseSize = 100
LearningRate1 = 0.00004
LearningRate2 = 0.0004
Kernel_Size = (5, 5)

tf.reset_default_graph ()
X = tf.placeholder(tf.float32, [None, 128, 128, 3])
Z = tf.placeholder(tf.float32, [None, NoiseSize])
IsTraining = tf.placeholder(tf.bool)
dropout_rate = tf.placeholder(tf.float32)

DiscGlobalStep = tf.Variable(0, trainable = False, name = "DiscGlobal")
GeneGlobalStep = tf.Variable(0, trainable = False, name = "GeneGlobal")
 
Fake = Build_Generator(Z)
DiscReal = Build_Discriminator(X)
DiscGene = Build_Discriminator(Fake, True)

print (np.shape(Fake), np.shape(DiscReal), np.shape(DiscGene))
 
    
LossDiscReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits                              (logits=DiscReal, labels=tf.ones_like(DiscReal)))
LossDiscGene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits                              (logits=DiscGene, labels=tf.zeros_like(DiscGene)))

LossDisc = LossDiscReal + LossDiscGene
LossGene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits                          (logits=DiscGene, labels=tf.ones_like(DiscGene)))

DiscVars = tf.get_collection             (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Discriminator_Variable")
GeneVars = tf.get_collection             (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Generator_Variable")
UpdateOps = tf.get_collection (tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(UpdateOps):
    TrainDisc = tf.train.AdamOptimizer(LearningRate1, beta1 = 0.5)                             .minimize(LossDisc, var_list=DiscVars)
    TrainGene = tf.train.AdamOptimizer(LearningRate2, beta1 = 0.5)                             .minimize(LossGene, var_list=GeneVars)


Discriminator_Loss_Graph = []
Generator_Loss_Graph = []

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    TotalBatch = int(len(DirectoryList) / BatchSize)
    
    for epoch in range(TotalEpoch):
        
        LossDiscVal = 0 
        LossGeneVal = 0
        print('Epoch:', '%03d  ' %(epoch+1))

        for i in range(TotalBatch):
            
            BatchImage = Build_Shuffle_BatchData (BatchSize, SimpsonImage)
            NoiseVector = Build_GetNoise(BatchSize, NoiseSize)
            
            sess.run(TrainDisc, feed_dict = {X: BatchImage, Z: NoiseVector, IsTraining: True})
            LossDiscVal = sess.run(LossDisc, feed_dict = {X: BatchImage, Z: NoiseVector, IsTraining: True})
            
            sess.run(TrainGene, feed_dict = {X: BatchImage, Z: NoiseVector, IsTraining: True})
            LossGeneVal = sess.run(LossGene, feed_dict = {X: BatchImage, Z: NoiseVector, IsTraining: True})

        
        print('Discriminator loss: {:.8}  '.format(LossDiscVal))
        print('Generator loss:     {:.8}  '.format(LossGeneVal))

        Discriminator_Loss_Graph.append(LossDiscVal)
        Generator_Loss_Graph.append(LossGeneVal)
        
        NoiseVector = Build_GetNoise(10, NoiseSize)
        Samples1 = sess.run(Fake, feed_dict = {Z: NoiseVector, IsTraining: False})

        fig, ax = plt.subplots(1, 5, figsize=(30, 30))

        for i in range(5) :
            ax[i].imshow(Samples1[i])

        plt.savefig("samples/{}.png".format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

        if (epoch+1) % 50 == 0:
            saver.save(sess, "C:/Users/Maritimus/Desktop/Simpson DCGAN/Simpson/save/DCGAN_save_" + str(epoch+1) + ".ckpt")
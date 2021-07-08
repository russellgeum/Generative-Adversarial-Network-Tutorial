import os, ssl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from module import BEGAN
from utility import Data_Processing
from utility import GetNoiseFunctions

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context
    
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

TrainImage = mnist.train.images
TrainImage = np.reshape (TrainImage, (-1, 28, 28 ,1))
print (np.shape(TrainImage))



tf.reset_default_graph()

Gamma = 1.25
Lambda = 0.01

TotalEpoch= 100
BatchSize = 128
NoiseSize = 100
kernel_size = (3, 3)

BEGANs = BEGAN(32, kernel_size)
Shuffles = Data_Processing()
GetNoise = GetNoiseFunctions()

GeneInput = tf.placeholder(tf.float32, [None, NoiseSize])
DiscInput = tf.placeholder(tf.float32, [None, 28, 28, 1])
Kappa = tf.Variable(0., trainable = False)

DiscGlobalStep = tf.Variable(0, trainable = False, name = "DiscGlobal")
GeneGlobalStep = tf.Variable(0, trainable = False, name = "GeneGlobal")

Fake = BEGANs.Generator(GeneInput)
DiscReal, RealLoss = BEGANs.Discriminator(DiscInput, 2)
DiscFake, FakeLoss = BEGANs.Discriminator(Fake, 2, True)

RealLoss = RealLoss / BatchSize
FakeLoss = FakeLoss / BatchSize

print (DiscReal.shape)
print (DiscFake.shape)
print (Fake.shape)


# Caculate Loss Function, Session에서 sess.run (~~)
# Note!!!, kappa(t+1) = kappa(t) + Lambda*(Gamma*RealLoss - FakeLoss)
# Disc의 전체 손실이 아니라, 오직 진짜 이미지에 대한 Disc 손실로 kappa를 갱신
DiscLoss = RealLoss - Kappa * FakeLoss
GeneLoss = FakeLoss
UpdateKappa = Kappa.assign (tf.clip_by_value (Kappa + Lambda*(Gamma*RealLoss-FakeLoss), 0, 1))
print (type(UpdateKappa))
print (type(Kappa))


# tf.get_collection과 tf.GraphKeys.TRAINABLE_VARIABLES with "scope 이름" 을 이용
# Discriminator의 변수와 Generator의 변수는 따로 학습
# tf.control_dependecies는 묶음 연산과 실행 순서를 정의하는 메서드
# UpdataOps를 먼저 실행하고 TrainDisc, TrainGene을 실행
DiscVars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "DiscrimnatoScope")
GeneVars = tf.get_collection (tf.GraphKeys.TRAINABLE_VARIABLES, scope = "GeneratorScope")
UpdateOps = tf.get_collection (tf.GraphKeys.UPDATE_OPS)


with tf.control_dependencies (tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    TrainDisc = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)\
                .minimize(DiscLoss, var_list = DiscVars, global_step = DiscGlobalStep)
    TrainGene = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1 = 0.5)\
                .minimize(GeneLoss, var_list = GeneVars, global_step = GeneGlobalStep)

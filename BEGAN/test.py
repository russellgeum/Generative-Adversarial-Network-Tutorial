import os, ssl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model
from Module import BEGAN
from Utility import Data_Processing
from Utility import GetNoiseFunctions
from Utility import LatentSpace



tf.reset_default_graph()

BEGANs = BEGAN(32, (3, 3))
LatentSpace = LatentSpace(10, 100)
GetNoise = GetNoiseFunctions()

GeneInput = tf.placeholder(tf.float32, [None, 100])
Fake = BEGANs.Generator(GeneInput)



Latent = LatentSpace.Build_LatentSpace()
print (np.shape(Latent))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore (sess, "./save/model.ckpt-50")

    Result3 = sess.run (Fake, feed_dict = {GeneInput : Latent})

    fig, ax = plt.subplots(1, 10, figsize = (20, 20))

    for i in range(10):
        ax[i].imshow(np.reshape(Result3[i], (28, 28)))

    plt.show ()
    plt.close(fig)

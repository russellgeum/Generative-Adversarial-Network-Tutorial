import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *

import numpy as np
import matplotlib.pyplot as plt

"""
tf.keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

tf.keras.layers.Conv2DTranspose(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    output_padding=None,
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

tf.reshape(
    tensor,
    shape,
    name=None
)
"""


# In[2]:


tf.__version__
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[7]:


class LSGAN_model ():
    
    def __init__(self):
        
        # MNIST image size, channels -> shape && noise_dimenstion size
        self.image_rows = 28
        self.image_cols = 28
        self.image_channels = 1
        self.image_shape = (self.image_rows, self.image_cols, self.image_channels)
        self.noise_dimension = 100
        
        # Convolution optional turpel for LSGAN architecture
        self.filter1, self.filter2, self.filter3, self.filter4 = 128, 64, 32, 1
        self.two_strides = (2, 2)
        self.one_strides = (1, 1)
        self.kernel_size = (3, 3)
        
        self.noise_dimenstion = 100
        self.learning_rate = 2e-4
        
        
    def build_Generator(self, inputs):
        
        """
        # Convolution optional turpel for LSGAN architecture
        self.filter1, self.filter2, self.filter3, self.filter4 = 128, 64, 32, 1
        self.two_strides = (2, 2)
        self.one_strides = (1, 1)
        self.kernel_size = (3, 3)
        """
        dense_size = 7*7*128
        dense_size_turple = (-1, 7, 7, 128)
        
        # batch_size에 대해서는 입력하지 않아도 자동으로 none
        generator = Dense(dense_size, activation = None) (inputs)
        generator = tf.reshape(generator, dense_size_turple)
        # generator = BatchNormalization() (generator)
        generator = relu(generator)
        
        ## 2
        generator = Conv2DTranspose(
                    filters = 128,
                    kernel_size = self.kernel_size,
                    strides = self.two_strides, 
                    padding = "same") (generator)
        # generator = BatchNormalization() (generator)
        generator = relu(generator)
        
        generator = Conv2DTranspose(
                    filters = 32,
                    kernel_size = self.kernel_size,
                    strides = self.two_strides, 
                    padding = "same") (generator)
        # generator = BatchNormalization() (generator)
        generator = relu(generator)
        
        generator = Conv2DTranspose(
                    filters = 1,
                    kernel_size = self.kernel_size,
                    strides = self.one_strides, 
                    padding = "same") (generator)
        generator = tanh(generator)
        
        outputs = generator
        
        return outputs
    
    
    def build_Discriminator(self, inputs):
        
        """
        # Convolution optional turpel for LSGAN architecture
        self.filter1, self.filter2, self.filter3, self.filter4 = 128, 64, 32, 1
        self.two_strides = (2, 2)
        self.one_strides = (1, 1)
        self.kernel_size = (3, 3)
        """
        discriminator = Conv2D(
                        filters = self.filter3,
                        kernel_size = self.kernel_size,
                        strides = self.two_strides,
                        padding = "same") (inputs)
        discriminator = tf.nn.leaky_relu(discriminator)
        
        discriminator = Conv2D(
                        filters = self.filter2,
                        kernel_size = self.kernel_size,
                        strides = self.two_strides,
                        padding = "same") (discriminator)
        # discriminator = BatchNormalization() (discriminator)
        discriminator = tf.nn.leaky_relu (discriminator)
        
        discriminator = Conv2D(
                        filters = self.filter1,
                        kernel_size = self.kernel_size,
                        strides = self.two_strides,
                        padding = "same") (discriminator)
        # discriminator = BatchNormalization() (discriminator)
        discriminator = tf.nn.leaky_relu (discriminator)
        
        discriminator = Flatten()(discriminator)
        discriminator = Dense(1, activation = None) (discriminator)
        
        outputs = discriminator
        
        return outputs
    
    def train_FUNCTION(self, epochs = 100, batch_size = 128):
        
        
        # Create discriminator Model
        discriminator_inputs = Input (shape = (28, 28, 1))
        discriminator_outputs = self.build_Discriminator(discriminator_inputs)
        DISCRIMINATOR = Model (inputs = discriminator_inputs, 
                               outputs = discriminator_outputs,
                               name = "DISCRIMIANTOR")
        DISCRIMINATOR.compile(optimizer = RMSprop(learning_rate = self.learning_rate, 
                                                  rho = 0.5),
                              loss = 'mse',
                              metrics = ["accuracy"])
        DISCRIMINATOR.summary()

        # Create generator model, But Do not compile
        generator_inputs = Input (shape = (self.noise_dimenstion,))
        generator_outputs = self.build_Generator(generator_inputs)
        GENERATOR = Model(inputs = generator_inputs, 
                          outputs = generator_outputs, 
                          name = "GENERATOR")
        GENERATOR.summary()
        
        # Last, combine discriminator + generator model.
        # The model is not trained for all layers, only generator model
        # So, We need discriminator of trainiable method. 
        entired_inputs = GENERATOR(generator_inputs)
        entired_outputs = DISCRIMINATOR (entired_inputs)
        LSGAN_MODEL = Model (inputs = generator_inputs, outputs = entired_outputs,
             name = "LSGAN_MODEL")
        LSGAN_MODEL.compile(optimizer = RMSprop(learning_rate = 0.5*self.learning_rate, 
                                                rho = 0.5),
                            loss = 'mse', 
                            metrics = ["accuracy"])
        LSGAN_MODEL.summary()
        
        
        """
        1. Load MNIST Data
        2. "real" is target value of real image
        3. "fake" is target value of fake image
        """
        (train_image, _), (_, _) = mnist.load_data()
        train_image = train_image / 255.
        real, fake = np.ones((batch_size, 1)), np.zeros((batch_size, 1))
        
        # 1 epoch = 60,000 images
        for epoch in range (epochs):
            
            for batch in range (int(60000/batch_size)):

                # Select a random batch of images
                select_index = np.random.randint(0, train_image.shape[0], batch_size)
                select_image = train_image[select_index]
                select_image = np.reshape(select_image, (-1, 28, 28, 1))

                # random_noise를 발생시키고, Pick a generator_images
                random_noise = np.random.normal(0., 1., size = ([batch_size, self.noise_dimenstion]))
                generator_image = GENERATOR.predict(random_noise)

                """
                Select a random batch of image
                ---> np.random.randint(0, 60000, 128) : 0에서 60000 사이 중, 128개를 랜덤으로 고르는 것
                """
                # Train the discriminator
                # 진짜 이미지가 들어오면 1로 판단, 가짜 이미지가 들어오면 0으로 판단하게끔 훈련
                discriminator_loss_real = DISCRIMINATOR.train_on_batch(select_image, real)
                discriminator_loss_fake = DISCRIMINATOR.train_on_batch(generator_image, fake)
                discriminator_loss = 0.5*(np.add(discriminator_loss_real, discriminator_loss_fake))

                # Train the generator
                # random_noise를 넣어서 가짜 이미지가 진짜 이미지처럼 보이기 위해 valid_target = 1이 되도록 학습
                DISCRIMINATOR.trainable = False
                generator_loss = LSGAN_MODEL.train_on_batch(random_noise, real)
                generator_loss = 0.5 * np.float32(generator_loss)

                print ("%d %d [Disc loss: %f, acc.: %.2f%%] [Gene loss: %f, acc.: %.2f%%]" \
                       %(epoch+1, batch+1, discriminator_loss[0], 100*discriminator_loss[1], 
                         generator_loss[0], 100*generator_loss[1]))


                test_noise = np.random.normal (0., 1., (10, self.noise_dimenstion))
                test_generator = GENERATOR.predict (test_noise)
                test_generator = test_generator * 0.5 + 0.5
            
            
            """
            Show fake images per epoch
            """
            fig, ax = plt.subplots(1, 10, figsize = (20, 10))
            for i in range(10):

                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(test_generator[i], (28, 28)))

            plt.show ()
            plt.close(fig)



if __name__ == '__main__':
    LSGAN = LSGAN_model()
    LSGAN.train_FUNCTION()

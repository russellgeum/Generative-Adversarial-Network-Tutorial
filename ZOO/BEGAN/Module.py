import tensorflow as tf
import numpy as np


class BEGAN():
    
    def __init__ (self, filters, kernel_size):

        """ex) 8, 16, 32"""
        self.kernel = kernel_size
        self.filter1 = 1*filters
        self.filter2 = 2*filters
        self.filter3 = 3*filters

        """Arg:
        kernel_size : 커널 사이즈를 입력한다. 
        입력 커널의 형태는 정수 튜플 쌍으로 들어와야 한다.
        filters : 정수 값을 입력받고, 값의 배수마다 새로운 필터수로 할당
        (ex, 8을 입력하면 8 16 32)"""

    # 인코더로 들어오는 이미지의 크기는 28, 28, 1이다.
    def Encoder_Module (self, inputs):

        outputs = tf.layers.conv2d(inputs = inputs, 
                                filters = self.filter1, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter2, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)

        outputs = tf.layers.max_pooling2d(inputs = outputs, 
                                pool_size = self.kernel, 
                                strides = (2, 2), 
                                padding = "same")
        # 여기로 들어온 이미지의 크기는 14, 14, slef.filter1으로 다운

        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter2, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter3, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.max_pooling2d(inputs = outputs, 
                                pool_size = self.kernel, 
                                strides = (2, 2), 
                                padding = "same")
        # 여기로 들어온 이미지의 크기는 7, 7, slef.filter1으로 다운
        

        outputs = tf.reshape(outputs, [-1, 7*7*self.filter3])
        outputs = tf.layers.dense(outputs, 100)
        outputs = tf.nn.tanh(outputs)

        return outputs

    # 디코더로 들어오는 입력의 크기는 100이다.
    def Deconder_Module (self, inputs):

        # noise dimension is 100, so form 100 to 7*7*f3 dense netwrok
        outputs = tf.layers.dense(inputs, 7*7*self.filter1)
        outputs = tf.nn.relu(outputs)
        outputs = tf.reshape(outputs, [-1, 7, 7, self.filter1])

        """여기까지 7, 7, f1 크기"""
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter1, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter1, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.image.resize(outputs, [14, 14])

        """여기까지 14, 14, f1 크기"""
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter2, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.conv2d(inputs = outputs, 
                                filters = self.filter1, 
                                kernel_size = self.kernel, 
                                strides = (1, 1), 
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.image.resize(outputs, [28, 28])

        """여기까지 28, 28, f2 크기"""
        outputs = tf.layers.conv2d(inputs = outputs,
                                filters = self.filter1,
                                kernel_size = self.kernel,
                                strides = (1, 1),
                                padding = "same")
        outputs = tf.nn.elu(outputs)
        outputs = tf.layers.conv2d(inputs = outputs,
                                filters = 1,
                                kernel_size = self.kernel,
                                strides = (1, 1),
                                padding = "same")
        outputs = tf.nn.tanh(outputs)


        """출력 크기는 28, 28, 1로 원본 이미지랑 동일"""
        return outputs

    def Generator (self, inputs, reuse = None):
        with tf.variable_scope("GeneratorScope") as scope:
            if reuse:
                scope.reuse_variables()
            
            outputs = self.Deconder_Module(inputs)
        
        return outputs

    def Discriminator (self, inputs, norm_species, reuse = None):
        with tf.variable_scope("DiscrimnatoScope") as scope:
            if reuse:
                scope.reuse_variables()
            
            outputs = self.Encoder_Module(inputs)
            outputs = self.Deconder_Module(outputs)
            try:
                if norm_species == 1:
                    Norm = tf.reduce_mean (tf.abs(outputs - inputs))

                elif norm_species == 2:
                    Norm = tf.sqrt(2*tf.nn.l2_loss(outputs - inputs))

                else:
                    raise Exception("L2 노름 내지, L1 노름만 구할 것입니다. \
                                    1 또는 2를 입력해주세요.")
            except Exception as error:
                print ("오류가 발생하였어요.", error)
    
        
        return outputs, Norm



if __name__ == "__main__":
    tf.reset_default_graph()
    DiscInput = tf.placeholder(tf.float32, [None, 28, 28, 1])
    BEGANs = BEGAN(8, (3, 3))
    result1, result2 = BEGANs.Discriminator (DiscInput, 2, False)

    print (np.shape(result1))
    print (np.shape(result2))

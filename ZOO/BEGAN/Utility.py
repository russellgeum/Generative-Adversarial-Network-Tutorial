import numpy as np
import tensorflow as tf



class Data_Processing():

    # inputs은 넘파이의 array 형태일 것, 그래야 shuffle이 가능
    def Build_ShuffleData (self, inputs, batch_size) :
        
        np.random.shuffle (inputs)
        temp_image = inputs [0 : batch_size]
        outputs = temp_image
        
        return outputs

    def Build_SliceData (self, inputs, new_size):

        temp_data = inputs[:new_size]
        outputs = temp_data

        return outputs



class GetNoiseFunctions():
    
    # 노이즈를 발생하는 넘파이 함수들이 모인 곳, 취향따라 선택이 가능
    def Build_UniformNoise (self, batch_size, noise_size):
        outputs = np.random.uniform(-1.0000, 1.0000, size = [batch_size, noise_size])
        return outputs

    def Build_Twice_UniformNoise (self, batch_size, noise_size):
        outputs = np.random.uniform(-2.0000, 2.0000, size = [batch_size, noise_size])
        return outputs

    def Build_Half_UniformNoise (self, batch_size, noise_size):
        outputs = np.random.uniform(+0.0000, 1.0000, size = [batch_size, noise_size])
        return outputs

    def Build_GaussianNoise (self, batch_size, noise_size):
        outputs = np.random.normal(-1.0000, 1.0000, size = [batch_size, noise_size])
        return outputs

    def Build_Twice_GaussianNoise (self, batch_size, noise_size):
        outputs = np.random.normal(-2.0000, 2.0000, size = [batch_size, noise_size])
        return outputs



class LatentSpace():

    def __init__ (self, delta, NoiseLength):

        self.delta = delta
        self.NoiseLength = NoiseLength

    def Build_LatentSpace (self):
    
        StartArray = np.random.uniform(-1.00, 1.00, size = [1, self.NoiseLength])
        EndinArray = np.random.uniform(-1.00, 1.00, size = [1, self.NoiseLength])
        DeltaArray = (StartArray - EndinArray)/self.delta

        LatentSpaceList = []
        for index in range (self.delta):
            StartArray = StartArray + DeltaArray
            LatentSpaceList.append(StartArray)
            
        LatentSpaceList = np.reshape(LatentSpaceList, (self.delta, 100))

        return LatentSpaceList

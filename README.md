# Introdunction  
MNIST 데이터들로 여러가지 Generative Adversarial Networks을 구현  
논문의 조건을 생각하고, 더 나아가 바꾸어가면서 GAN 알고리즘을 연습  
그리고 심슨 이미지와 포켓몬 이미지 데이터로 DCGAN을 구현 (포켓몬 이미지 코드는 증발)  

# DCGAN for Simpson
### Timeline
![Time Line](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_simpson/timeline.gif)
### Latent Space
![DCGAN Latent](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_simpson/latent.gif)

# DCGAN for Pokemon  
### High Resolution  
![High1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_pokemon1/high%20(7).png)
![High2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_pokemon1/high%20(6).png)
![High3](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_pokemon1/high%20(4).png)


# GAN Zoo for MNIST  
## 1. GAN  
- Generative Adversairal Network의 첫 시작이 되는 논문  

![DCGAN1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_dcgan/DCGAN1.png)
![DCGAN2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_dcgan/DCGAN2.png)
![DCGAN3](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_dcgan/DCGAN3.png)

## 2. EBGAN  
- Generator의 수렴 속도가 매우 빨랐음  
- without sqrt에서는 81 Epoch부터 모델이 완전히 죽어버렸다. 왜 그럴까? sqrt가 빠져서 균형을 맞추지 못한 것 (?)    
- 논문에서의 내용대로 L2 Norm을 적용하면, 수렴 속도는 훨씬 느렸지만, 더 깔끔한 이미지를 얻을 수 있음  
### 2-1. EBGAN (without Square root)  
![EBGAN1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/EBGAN1.png) 
![EBGAN2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/EBGAN2.png) 
![EBGAN3](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/EBGAN3.png) 
### 2-2. EBGAN Model Collapse (without Square root)   
![Collapse](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/81%20epoch.png)
### 2-3. EBGAN with L2 Normalization  
![L21](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/L2%20Norm%201.png)
![L22](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/L2%20Norm%202.png)
![L23](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_ebgan/L2%20Norm%203.png)
  
## 3. BEGAN
- Gamma 값은 model의 diversiy를 결정. gamma 파라미터를 1 이하로 하면 심각한 mode collapse가 발생  
- 값을 1 이상으로 높이면 mode collapse를 조금 완화. 그러나 생성 이미지의 품질은 하락  
- Latent space를 탐색하면 GAN의 학습 특성이 기억 기반이 아님을 알 수 있음    
- GAN의 벤치마크로 쓰이는 데이터셋은 CelebA와 같이 MNIST에 비해 데이터 분포가 Continuous한 특성  
- BEGAN의 논문에서도 강한 model collapse에 대해서는 구체적인 대안을 제시하지 않음  
- 예로 백인 남성, 황인 남성, 흑인 남성의 데이터와 숫자 1, 2, 3의 데이터에서 어떤 차이가 있는지 알아야 함  
- Mode Collapse는 데이터 분포가 연속적인지, 이산적으로 분리되어 있는지에 민감한 의존성이 있을 것  
- BEGAN의 Mode collapse Escaping에 대한 논문은 존재 (앞으로 실험 예정)  
### 3-1. BEGAN (== 50 epochs)  
![BEGAN1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_began/BEGAN%20sample%201.png)
![BEGAN2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_began/BEGAN%20sample%202.png)
### 3-2. BEGAN Latent Space  
![LATENT1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_began/Latent%20Space%201.png)
![LATENT2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_began/Latent%20Space%202.png)
### 3-3. BEGAN Mode Collapse (75 epochs ~)  
![collapse](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_began/Mode%20collapse%202%20(75epoch).png)
  
## 4. WGAN  
- Loss Function은 reduce_mean으로 구현한 것이 차이점  
  
![WGAN1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_wgan/WGAN1.png)
![WGAN2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_wgan/WGAN2.png)
![WGAN3](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_wgan/WGAN3.png)

## 5. LSGAN  
- 나의 구현에서는 Adam optimizer는 좋은 성능을 보이지 못함 (batch nomalization을 적용하였음에도 불구하고)  
- 적절한 learning rate를 찾는데 오래 걸렸음 (이전에 구현했던 MNIST GAN을 기반으로 헀을때)  
- RMSProp (learning_rate = 0.0005 for discriminator, learning_rate = 0.00025 for generator)를 권장  
- 60,000장 이미지를 모두 사용했을 때 1 epoch 이면 최소 20~25 epoch 이상을 권장  
![LSGAN1](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_lsgan/LSGAN%201.png)
![LSGAN2](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_lsgan/LSGAN%202.png)
![LSGAN3](https://github.com/Doyosae/Generative_Adversarial_Network/blob/master/sample/sample_lsgan/LSGAN%203.png)

# Reference  
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126  )
- [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
- [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/abs/1703.10717)
- [Escaping from Collapsing Modes in a Constrained Space](https://arxiv.org/abs/1808.07258)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)  
- [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)

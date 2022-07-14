# NEURAL NETWORK FRAGILE WATERMARKING WITH NO MODEL PERFORMANCE DEGRADATION
This code is the implementation of the fragile neural network watermarking method introduced in the paper "Neural network fragile watermarking with no model performance degradation".

## Abstract

Deep neural networks are vulnerable to malicious fine-tuning attacks such as data poisoning and backdoor attacks. Therefore, in recent research, it is proposed how to detect malicious fine-tuning of neural network models. However, it usually negatively affects the performance of the protected model. Thus, we propose a novel neural network fragile watermarking with no model performance degradation. In the process of watermarking, we train a generative model with the specific loss function and secret key to generate triggers that are sensitive to the fine-tuning of the target classifier. In the process of verifying, we adopt the watermarked classifier to get labels of each fragile trigger. Then, malicious fine-tuning can be detected by comparing secret keys and labels. Experiments on classic datasets and classifiers show that the proposed method can effectively detect model malicious fine-tuning with no model performance degradation.

## 摘要

深度神经网络容易受到数据中毒和后门攻击等恶意微调攻击。 因此，在最近的研究中，提出了如何检测神经网络模型的恶意微调。 但是，它通常会对目标模型的性能产生负面影响。 因此，我们提出了一种新型神经网络脆弱水印技术，不会影响模型的性能。 在水印的过程中，我们训练了一个具有特定损失函数和密钥的生成模型，以生成对目标分类模型微调变化敏感的触发集。 在验证过程中，我们用目标水印分类模型得到每个脆弱触发集的标签。 然后，通过比较密钥和标签的差异来检测恶意微调。 在经典数据集和分类模型上的实验表明，该方法可以有效地检测模型是否被恶意微调，而不会降低模型性能。

## How to use the code

There are two parts:

  1.put the model you want to watermark in `./train_model`
  
  2.run `train.py`

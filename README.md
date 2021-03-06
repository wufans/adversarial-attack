# Adversarial Example Tests
This respository gives some test results of adversarial examples generation algorithms such as **FGSM, CW, Deepfool and JSMA** ,and we also print output of the last layer to exploit possible adversarial example detecting method.

# Dependencies
If you want to test these projects, following dependencies you must need:
  
  - python3.5
  - tensorflow

  
Installing TensorFlow will take care of all other dependencies like **numpy**，**keras** and **scipy**.
# Tutorials
 ## deepfool
 The target models of this algorithms has been pretrained, by programing:
 ```python
 models.resnet34(pretrained=True)
 ```
 return a pretrained models with ImageNet datasets.
 Or you can train a model yourself. In our work, we train a classification models with MNIST datasets.
 ## JSMA
 This part we cite the work of [Papernot et al.](https://github.com/tensorflow/cleverhans).
 Default model in the source code is a deep neural network defined in above respository.
 This part relies on cleverhans's other files, you my need to install the whole respository for running this code.
 ## FGSM
 We also cite this work from [cleverhans](https://github.com/tensorflow/cleverhans).This tutorial covers how to train a MNIST/CIFAR model using TensorFlow, craft adversarial examples using the fast gradient sign method, and make the model more robust to adversarial examples using adversarial training.
 ## CW attack
 CW attack consists of L0 attack,L2 attack and Li attack. In our work, we only test L2 attack.This tutorial covers how to train a          MNIST model using TensorFlow, craft adversarial examples using CW attack. More details in [C&W attack](https://github.com/tensorflow/cleverhans).

# Reference code

[cleverhans](https://github.com/tensorflow/cleverhans)

# Cite our work
Min F, Qiu X, Wu F. Adversarial attack? Don't panic[C]//2018 4th International Conference on Big Data Computing and Communications (BIGCOM). IEEE, 2018: 90-95.

# Reference papers
[1]. Goodfellow I J, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples[J]. Computer Science, 2014.

[2]. Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. 2016.

[3].Szegedy, Christian, Zaremba, Wojciech, Sutskever, Ilya, Bruna, Joan, Erhan, Dumitru, Goodfellow, Ian J., and Fergus, Rob. Intriguing properties of neural networks.CoRR, abs/1312.6199, 2013.

[4]. guyen, A., J. Yosinski, and J. Clune, Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. 2015: p. 427-436.

[5]. Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard. “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks”. Conference on Computer Vision and Pattern Recognition (CVPR) 2016.

[6].Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, and Ananthram Swami. “The Limitations of Deep Learning in Adversarial Settings”. IEEE European Symposium on Security and Privacy (Euro S&P) 2016.

[7]. Mopuri K R, Garg U, Babu R V. Fast Feature Fool: A data independent approach to universal adversarial perturbations[J]. 2017.

[8]. Nicholas Carlini and David Wagner. 2017. Towards Evaluating the Robustness of Neural Networks. In IEEE Symposium on Security and Privacy (Oakland) 2017.

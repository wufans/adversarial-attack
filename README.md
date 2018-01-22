# adversarial-attack
This respository gives some test results of adversarial examples generation algorithms such as **FGSM, CW, Deepfool, JSMA and Fast feature fool** ,and we also print output of the last layer to exploit possible adversarial example detecting method.

# Dependencies
If you want to test these projects, following dependencies you must need:
  
  -python3.5
  -tensorflow
  -torch
Installing TensorFlow and torch will take care of all other dependencies like **numpy** and **scipy**.
# Steps for running source code
 ## deepfool
 The target medels of this algorithms has been pretrained, by announcing:
 ```python
 models.resnet34(pretrained=True)
 ```
 return a pretrained medels in ImageNet dataset.
 or you can train a medel yourself. In our work, we train a classification models with MNIST datasets.
 ## JSMA
 This part we cite the work of [Papernot et al.](https://github.com/tensorflow/cleverhans).
 Default model in the source code is a deep neural network defined in above respository.
# Reference code

[cleverhans](https://github.com/tensorflow/cleverhans)

[deepfool](https://github.com/LTS4/DeepFool)

# Reference papers
[1]. Goodfellow I J, Shlens J, Szegedy C. Explaining and Harnessing Adversarial Examples[J]. Computer Science, 2014.
[2]. Kurakin A, Goodfellow I, Bengio S. Adversarial examples in the physical world[J]. 2016.
[3].Szegedy, Christian, Zaremba, Wojciech, Sutskever, Ilya, Bruna, Joan, Erhan, Dumitru, Goodfellow, Ian J., and Fergus, Rob. Intriguing properties of neural networks.CoRR, abs/1312.6199, 2013.
[4]. guyen, A., J. Yosinski, and J. Clune, Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. 2015: p. 427-436.
[5]. Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard. “DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks”. Conference on Computer Vision and Pattern Recognition (CVPR) 2016.
[6].Nicolas Papernot, Patrick McDaniel, Somesh Jha, Matt Fredrikson, Z. Berkay Celik, and Ananthram Swami. “The Limitations of Deep Learning in Adversarial Settings”. IEEE European Symposium on Security and Privacy (Euro S&P) 2016.
[7]. Mopuri K R, Garg U, Babu R V. Fast Feature Fool: A data independent approach to universal adversarial perturbations[J]. 2017.

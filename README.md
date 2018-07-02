# softmax_variants
Various loss functions about variants of softmax: center loss, cosface loss, large-margin gaussian mixture, COCOLoss
implemented by pytorch 0.3.1

the training dataset is MNIST

You can directly run code train_mnist_xxx.py to reproduce the result

The reference papers are as follow:

Center loss: Yandong Wen, Kaipeng Zhang, Zhifeng Li and Yu Qiao. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016

Cosface loss: Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu. CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018

Large-margin gaussian mixture loss: Weitao Wan, Yuanyi Zhong,Tianpeng Li, Jiansheng Chen. Rethinking Feature Distribution for Loss Functions in Image Classification. CVPR 2018

COSO loss: Yu Liu, Hongyang Li, Xiaogang Wang. Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017

The learned 2-d embedding features are:

softmax loss

![image](https://github.com/YirongMao/softmax_variants/blob/master/images/softmax_loss_epoch%3D50.jpg)

COCO loss

![image]( https://github.com/YirongMao/softmax_variants/blob/master/images/coco_loss_epoch%3D50.jpg)

Center loss

![image](https://github.com/YirongMao/softmax_variants/blob/master/images/center_loss_epoch%3D50.jpg)

CosFace loss

![image](https://github.com/YirongMao/softmax_variants/blob/master/images/LMCL_loss_u_epoch%3D50.jpg)

Large-margin gaussian mixture loss

![image](https://github.com/YirongMao/softmax_variants/blob/master/images/LGM_loss_epoch%3D50.jpg)
 



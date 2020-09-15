# SKINAI | Deep learning based web application for skin lesion classification 
 This repo is our solution for the final project of the course Distributed Programming and Networking of [MAIA](https://maiamaster.udg.edu/) master program.  
 The project is a client side web application for skin lesion classification with deep learning. 
## Application

![image](/media/uploadedimg/ss2.png)
![image](/media/uploadedimg/ss1.png)
## Model

![image](/media/uploadedimg/mobnet.png)
We used a lightweight CNN in this project ([MobileNet](https://www.semanticscholar.org/paper/MobileNets%3A-Efficient-Convolutional-Neural-Networks-Howard-Zhu/3647d6d0f151dc05626449ee09cc7bce55be497e)) because the model is deployed on client side. We use ImageNet initialization and perform selective sampling during the training.
![image](/media/uploadedimg/mod.png)

## Data
In this project we use [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset for training. For training we perform 80/20 percent training/validation split.



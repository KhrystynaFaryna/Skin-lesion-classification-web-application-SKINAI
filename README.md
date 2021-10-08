## SKINAI: a Deep learning based web application for skin lesion classification 
 This repo includes our (Aigerim Dautkulova, Khrystyna Faryna and Rua Khaled) solution for the final project of the course Distributed Programming and Networking of [MAIA](https://maiamaster.udg.edu/) master program.  
 The project is a client side web application for skin lesion classification with deep learning. 
 *This app is designed as a part of university course and is NOT meant for any kind of clinical use.*
## Application

![image](/media/uploadedimg/ss2.png)
![image](/media/uploadedimg/ss1.png)
## Model
![image](/media/uploadedimg/mod.png)
![image](/media/uploadedimg/mobnet.png)
We used a lightweight CNN in this project ([MobileNet](https://www.semanticscholar.org/paper/MobileNets%3A-Efficient-Convolutional-Neural-Networks-Howard-Zhu/3647d6d0f151dc05626449ee09cc7bce55be497e)) because the model is deployed on client side. We use ImageNet initialization and perform selective sampling during the training.


## Data
In this project we use [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dataset for training. For training we perform 80/20 percent training/validation split.



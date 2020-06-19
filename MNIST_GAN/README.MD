{\rtf1\ansi\ansicpg1252\cocoartf2511
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww12600\viewh7280\viewkind0
\deftab720
\pard\pardeftab720\ri0\partightenfactor0

\f0\fs24 \cf0 # MNIST_GAN\
Use an ensemble model to generate adversarial examples using Generative Adversarial network (GAN)\
\
## Essential Information\
\
1. GANS built implementing pytorch\
\
\
2. Effective training on CPU takes 12+ hours. For better performance, GPU is recommended. Enable Cuda via pytorch. \
https://pytorch.org/docs/stable/notes/cuda.html\
\
#### Training  MNIST Untargeted Attack \
```bash\
$ python3 train_gan.py --model Model_X \
```  \
#### Training MNIST_GAN Targeted Attack\
```bash\
$ python3 train_gan.py --model Model_C --target 4 --thres 0.3 \
# thres: Perturbation bound \
```  \
Use ```--help``` for other arguments available (```epochs```, ```batch_size```, ```lr``` etc.)\
\
\
#### Training Target Model (Model X)\
```bash\
$ python3 target_models.py --model Model_X\
```  \
\
For visualizations,\
```bash\
$ python3 generators.py\
$ python3 discriminators.py\
```  \
}
# CycleGAN
Implementing CycleGAN Using Python

Assuming you already have python installed

Installing dependencies on terminal
```
py -m pip install matplotlib
py -m pip install pandas
py -m pip install keras
```

Also install keras contribution, must have [GIT installed](https://medium.com/@kegui/how-to-install-keras-contrib-7b75334ab742): 
```
git clone https://github.com/keras-team/keras-contrib.git/
cd keras-contrib
py setup.py install
```


Must create a datasets folder, in side this any dataset that has to containt 4 folders: testA, testB, trainA, traingB. Dataset example can be downloaded [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)
```
\_ datasets\
     \_ monet2real\
           \_   testA\
           \_   testB\
           \_   trainA\
           \_   trainB\
```


Reference: [rubikscode](https://rubikscode.net/2019/02/11/implementing-cyclegan-using-python/)

Based on [junyanz](https://junyanz.github.io/CycleGAN/)

These 5 code files are for CIFAR10 dataset . The models are ResNet20 and ResNet56
Steps:(for ResNet20)
a)import os
os.makedirs("cifarmodel", exist_ok=True)
b)python cifar_pretrain.py -l 20 --save_dir ./cifarmodel --epochs 30 --batch-size 128 --lr 0.1 --momentum 0.9 --wd 1e-4
c)mv cifarmodelresnet20.pkl cifarmodel/resnet20.pkl
##g = 2,3,4
d)python cifar_dsp.py -l 20 -g 4 -r 2e-3 
##g = 2,3,4
e)python cifar_finetune.py -l 20 -g 4 -p 0.6


Steps:(for ResNet56)
c)mv cifarmodelresnet20.pkl cifarmodel/resnet56.pkl
##g = 2,3,4
d)python cifar_dsp.py -l 56 -g 4 -r 2e-3 
##g = 2,3,4
e)python cifar_finetune.py -l 56 -g 4 -p 0.6

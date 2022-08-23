# GenErrorSSL_2022
Codes for "Information-Theoretic Characterization of the Generalization Error for Iterative Semi-Supervised Learning"

## Main codes
> bin_anchor_best.py AND bin_anchor_best_sgd.py: binary classfication on pairs of classes in CIFAR-10

> mul_anchor_best.py: multi-class classification on MNIST

## An example to run the experiment
```
CUDA_VISIBLE_DEVICES=0 python bin_anchor_best_sgd.py --exp_name bin_cifar10c19_sgd --classes 1 9 --num_labeled 500 --ratio_unlabeled 20 --SEED 111 \
                                                --tr_loops 100 --init_epochs 300 --init_batch_size 64 \
                                                --tr_epochs 20 --ratio_loop 5 --train_batch_size 64 --version 0 \
                                                --lr 5e-2 --wd 0 &
```

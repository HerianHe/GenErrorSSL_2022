import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16

import math
import numpy as np
    
    
list_list = [

    [
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v0/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v1/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v2/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v3/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v4/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v5/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v6/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v7/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v8/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v9/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v10/results.npy',
    'experiments/icml/bin_cifar10c78_lb500_m20_lp10_ep20_lr0.001_wd0.0_sd111_rl19_bs64_v11/results.npy',
    ],
    # [
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0_sd111_rl5_bs64_v0/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0_sd111_rl5_bs64_v1/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0_sd111_rl5_bs64_v2/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v4/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v5/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v7/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v9/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v10/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v11/results.npy',
    # 'experiments/icml/bin_cifar10c78_lb500_m20_lp100_ep20_lr0.001_wd0.0_sd111_rl5_bs64_v12/results.npy',
    # ],
    # [
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v0/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v1/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v2/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v3/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v4/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v9/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v10/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v11/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v12/results.npy',
    # 'experiments/icml/bin_cifar10c19_sgd_lb500_m20_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v13/results.npy',
    # ],
    # [
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v0/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v1/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v2/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v3/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v6/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v5/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v10/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v12/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v13/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v15/results.npy',
    # ],

    # [
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v0/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v2/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v3/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v4/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v6/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v7/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v16/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v10/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v12/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init250_ep20_lr0.05_wd0.0_sd111_rl5_bs64_v15/results.npy',
    # ],
    
    # [
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v0/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v1/results.npy',
    # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v2/results.npy',
    # ],
    
    # # [
    # # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd0_rl5_bs128_v0/results.npy',
    # # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd0_rl5_bs128_v1/results.npy',
    # # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd0_rl5_bs128_v2/results.npy',
    # # ],
    
    # # [
    # # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v0/results.npy',
    # # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v1/results.npy',
    # # 'experiments/icml/bin_cifar10c35_sgd_lb1000_m10_lp100_init300_ep20_lr0.05_wd0.0005_sd111_rl5_bs64_v2/results.npy',
    # # ],
    
    # [
    # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd111_rl5_bs128_v0/results.npy',
    # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd222_rl5_bs128_v1/results.npy',
    # 'experiments/icml/mul_mnist_res6_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0_sd333_rl5_bs128_v2/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd111_rl5_bs128_v3/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd222_rl5_bs128_v4/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd333_rl5_bs128_v5/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd555_rl5_bs128_v7/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd666_rl5_bs128_v8/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd777_rl5_bs128_v9/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd888_rl5_bs128_v10/results.npy',
    # 'experiments/icml/mul_mnist_res10_anchor_lb1000_m50_lp100_init200_ep20_lr0.001_wd0.0_sd888_rl5_bs128_v11/results.npy',
    # ],
    ]

flag_legend = 0

for file_list in list_list:
    test_acc_list = []
    test_loss_list = []
    train_acc_list = []
    train_loss_list = []
    gen_loss_list = []

    for i in range(len(file_list)):
        file = file_list[i]
        results = np.load(file)
        x = results[:,0]
        test_acc_list.append(results[:,1])
        test_loss_list.append(results[:,2])
        train_acc_list.append(results[:,3])
        train_loss_list.append(results[:,4])
        gen_loss_list.append(results[:,2]-results[:,4])


    test_acc_avg = np.array(test_acc_list).mean(axis=0) * 100
    test_acc_std = np.std(np.array(test_acc_list), axis=0) * 100
    
    test_loss_avg = np.array(test_loss_list).mean(axis=0)
    test_loss_std = np.std(np.array(test_loss_list), axis=0)
    
    train_acc_avg = np.array(train_acc_list).mean(axis=0) * 100
    train_acc_std = np.std(np.array(train_acc_list), axis=0) * 100
    
    train_loss_avg = np.array(train_loss_list).mean(axis=0)
    train_loss_std = np.std(np.array(train_loss_list), axis=0)
    
    gen_loss_avg = np.array(gen_loss_list).mean(axis=0)
    gen_loss_std = np.std(np.array(gen_loss_list), axis=0)
    
    # truncate
    iter2plot = 10
    x = x[:iter2plot]
    test_acc_avg = test_acc_avg[:iter2plot]
    test_acc_std = test_acc_std[:iter2plot]
    
    test_loss_avg = test_loss_avg[:iter2plot]
    test_loss_std = test_loss_std[:iter2plot]
    
    train_acc_avg = train_acc_avg[:iter2plot]
    train_acc_std = train_acc_std[:iter2plot]
    
    train_loss_avg = train_loss_avg[:iter2plot]
    train_loss_std = train_loss_std[:iter2plot]
    
    gen_loss_avg = gen_loss_avg[:iter2plot]
    gen_loss_std = gen_loss_std[:iter2plot]
    

    save_name = file_list[0].split('/')[-2].split('_v')[0]+'_avg'
    # save_name = 'test'

    w, h = figure.figaspect(2/3)
    print(w,h)
    fig = plt.figure(figsize=(w, h))

    ####
    # loss
    ax = fig.add_subplot(1, 1, 1,frameon=True)
    
    ax.margins(x=0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x, test_loss_avg, '--', color='r', alpha=0.8, linewidth=2, label='test_loss')
    ax.fill_between(x, np.squeeze(test_loss_avg - test_loss_std), np.squeeze(test_loss_avg + test_loss_std), color='r', alpha=0.2)
    
    ax.plot(x, train_loss_avg, '--', color='b', alpha=0.8, linewidth=2, label='train_loss')
    ax.fill_between(x, np.squeeze(train_loss_avg - train_loss_std), np.squeeze(train_loss_avg + train_loss_std), color='b', alpha=0.2)
    
    ax.plot(x, gen_loss_avg, '-', color='g', alpha=0.6, linewidth=2, label='gen_error')
    ax.fill_between(x, np.squeeze(gen_loss_avg - gen_loss_std), np.squeeze(gen_loss_avg + gen_loss_std), color='g', alpha=0.2)
    
    ax.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    plt.xticks(np.arange(0., iter2plot, 5))
    
    if flag_legend == 0:
        ax.legend( 
            loc='upper center', 
            # bbox_to_anchor=(.7, .75), 
            markerscale=0.3,
            fancybox=True, 
            # fontsize=10
            )
    
    plt.tight_layout()
    fig.savefig(os.path.join('./results/10times', save_name+'_loss_10iter.png'), dpi=200)
    plt.close()




    ######
    # accuray
    
    # fig = plt.figure()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(1, 1, 1, frameon=True)
    ax.margins(x=0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(x, test_acc_avg, '-', color='r', alpha=0.8, linewidth=2, label='test_acc')
    ax.fill_between(x, np.squeeze(test_acc_avg - test_acc_std), np.squeeze(test_acc_avg + test_acc_std), color='r', alpha=0.2)
    
    antt_init_x = 0
    ax.scatter(antt_init_x, test_acc_avg[antt_init_x], marker='o', color='r')
    antt_best_x = np.argmax(test_acc_avg)
    ax.scatter(antt_best_x, test_acc_avg[antt_best_x], marker='*', color='r')
    # ax.annotate(f'{test_acc_avg[antt_best_x]:.1f}%',  xy=(len(test_acc_avg)//2, test_acc_avg[antt_best_x]))
    ax.annotate(f'{test_acc_avg[antt_best_x]:.1f}%',  xy=(antt_best_x-6.5, test_acc_avg[antt_best_x]))
    
    ax.plot(x, train_acc_avg, '-', color='b', alpha=0.8, linewidth=2, label='train_acc')
    ax.fill_between(x, np.squeeze(train_acc_avg - train_acc_std), np.squeeze(train_acc_avg + train_acc_std), color='b', alpha=0.2)
    
    ax.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acc (%)')
    # plt.xticks(np.arange(0., iter2plot, 5))
    plt.xticks(np.arange(0., iter2plot, 5))
    
    if flag_legend == 0:
        ax.legend(
            loc='lower right',
            # bbox_to_anchor=(.7, .75), 
            markerscale=0.3,
            fancybox=True, 
            # fontsize=10
            )


    plt.tight_layout()
    fig.savefig(os.path.join('./results/10times', save_name+'_acc_10iter.png'), dpi=200)
    plt.close()
    
    flag_legend += 1
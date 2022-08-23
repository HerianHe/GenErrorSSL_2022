import random
import numpy as np
import logging
import os
import sys
import time
import pathlib
import math

# timer
class timer():
    def __init__(self):
        # print("Timer initialized @ "+ time.strftime('%Y-%m-%d-%H:%M:%S'))
        self.tic()
        
    def tic(self):
        self.t0 = time.time()
        # print(time.strftime('%Y-%m-%d-%H:%M:%S'))
        
    def toc(self, restart=True):
        diff = time.time()-self.t0
        if restart: self.t0 = time.time()
        return diff


# Network utils
# def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Argument Parsing
int2bool = lambda x: bool(int(x))

def str2list(isFloat, s, split='_'):
    l = s.split('_')
    if isFloat:
        l = [float(x) for x in l]
    else:
        l = [int(x) for x in l]
    return l

def str2bool(s):
    assert s in ['True', 'False']
    if s == 'True':
        return True
    else:
        return False
    
def get_input_output_dim(dataset):
    if dataset == 'CIFAR10' or dataset == 'SVHN':
        return 3, 10
    elif dataset == 'FMNIST' or dataset == 'MNIST':
        return 1, 10

# dir/path management
def makedirs(dirname):
    pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)

# logging
def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info('\n\n------ ******* ------ New Log ------ ******* ------')
    return logger


class get_epoch_logger():
    def __init__(self):
        self.epochs = []
        self.results = []
        self.best_epoch = 0; self.best_result = 0
        
    def append_results(self, list):
        self.epochs.append(list[0])
        self.results.append(list[1])
        self.update_best_epoch()
    
    def update_best_epoch(self):
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
    
    def update_best_epoch_to_logger(self, logger):
            
        if self.results[-1] >= self.best_result:
            self.best_epoch = self.epochs[-1]
            self.best_result = self.results[-1]
        
        logger.info('Best result @ {:03d}, {} \n'.format(self.best_epoch, self.best_result))
        return self.best_epoch


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams["font.family"] = "serif"

import math
import numpy as np
    
    
class log_test_train_acc_loss_history():
    def __init__(self, idx_indicator_for_best = 1, increasing=True):
        self.results = []
        self.idx = idx_indicator_for_best
        self.increasing=increasing
        self.best_timepoint = 0
            
    def append_results(self, list):
        self.results.append(list)
        self.update_best_timepoint()
    
    def update_best_timepoint(self):
        if self.increasing:
            if self.results[-1][self.idx] >= self.results[self.best_timepoint][self.idx]:
                self.best_timepoint = self.results[-1][0]
        else:
            if self.results[-1][self.idx] <= self.results[self.best_timepoint][self.idx]:
                self.best_timepoint = self.results[-1][0]
        self.best_test_acc = self.results[self.best_timepoint][1]
        self.best_test_loss = self.results[self.best_timepoint][2]
        self.best_train_acc = self.results[self.best_timepoint][3]
        self.best_train_loss = self.results[self.best_timepoint][4]
        
    def save_results(self, save_path):
        np.save(os.path.join(save_path, 'results.npy'), np.array(self.results))
        
    # from metric import topk_nat_acc_2
    # accuracy, avg_loss, confusion_matrix, classwise_acc_dict, classwise_loss_dict, classwise_margin_dict, message = topk_nat_acc_2(model.net, test_loader=test_loader)
    # write excel
    # columns = [str(i) for i in range(loader.num_classes)]
    # df = pd.DataFrame(confusion_matrix, columns=columns)
    # filename = args.checkpoint.split('/')[-3]
    # df.to_excel('./results/'+filename+'_confusion_matrix.xlsx')
    
    # df = pd.DataFrame(classwise_acc_dict, index=[0])
    # df.to_excel('./results/'+filename+'_classwise.xlsx')
    
    def visualize(self, save_name, verbose=False):
        results = np.array(self.results)
        steps, test_acc, test_loss, train_acc, train_loss, = results[:,0], results[:,1]*100, results[:,2], results[:,3]*100, results[:,4]       
        gen_loss = test_loss - train_loss

        f = plt.figure()
        ax = f.add_subplot(2, 1, 1, frameon=True)
        ax.margins(x=0.01)
        ax.spines['top'].set_visible(False)

        ln1 = ax.plot(steps, test_acc, '-', color='r', alpha=0.8, linewidth=1, label='test_acc')
        antt_init_x = 0
        ax.scatter(antt_init_x, test_acc[antt_init_x], marker='o', color='r')
        antt_best_x = np.argmax(test_acc)
        ax.scatter(antt_best_x, test_acc[antt_best_x], marker='*', color='r')
        ax.annotate(f'{test_acc[antt_best_x]:.1f}%',  xy=(len(test_acc)//2, test_acc[antt_best_x]))
        
        ln2 = ax.plot(steps, train_acc, '-', color='b', alpha=0.8, linewidth=1, label='train_acc')
        ax.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)
        ax.set_xlabel('Loop')
        
        ax_twin = ax.twinx()
        ax_twin.margins(x=0.01)
        ax_twin.spines['top'].set_visible(False)

        ln3 = ax_twin.plot(steps, test_loss, '--', color='r', alpha=0.8, linewidth=1, label='test_loss')
        ln4 = ax_twin.plot(steps, train_loss, '--', color='b', alpha=0.8, linewidth=1, label='train_loss')
        ln5 = ax_twin.plot(steps, gen_loss, '--', color='g', alpha=0.8, linewidth=1, label='gen_loss')
        ax_twin.grid(axis='x', color='grey', linestyle='-.', linewidth=0.5)

        leg = ln1 + ln2 + ln3 + ln4 +ln5
        labs = [l.get_label() for l in leg]
        ax.legend(leg, labs, 
                loc='lower right', 
                markerscale=1, fancybox=True, 
                )

        ax.set_ylabel('Acc (%)')
        ax_twin.set_ylabel('Loss')
        # plt.xticks(np.arange(0., 100, 5))
        
        if verbose:
            num_ulb = results[:,5] 
            ax = f.add_subplot(2, 1, 2, frameon=True)
            ax.plot(steps, num_ulb, '-', color='r', alpha=0.8, linewidth=1, label='num_ulb')
        
        
        plt.tight_layout()
        f.savefig(save_name, dpi=200)
        plt.close()
    
        
        
class log_single_result_history():
    def __init__(self, increasing=True):
        self.timepoints = []
        self.results = []
        self.increasing=increasing
        self.best_timepoint = 0
        if increasing:
            self.best_result = -1e8
        else:
            self.best_result = 1e8
            
    def append_results(self, list):
        assert len(list)==2
        self.timepoints.append(list[0])
        self.results.append(list[1])
        self.update_best_timepoint()
    
    def update_best_timepoint(self):
        if self.increasing:
            if self.results[-1] >= self.best_result:
                self.best_timepoint = self.timepoints[-1]
                self.best_result = self.results[-1]
        else:
            if self.results[-1] <= self.best_result:
                self.best_timepoint = self.timepoints[-1]
                self.best_result = self.results[-1]


import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure as figure
plt.rcParams["font.family"] = "serif"
# plt.rcParams['font.size'] = 12

def subplot_figure(root, file, data_plots):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = [subplot_1, subplit_2,]
    """
    # pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    assert isinstance(data_plots, list)
    num_plots = len(data_plots)
    w, h = figure.figaspect(1/2 * num_plots)
    fig = plt.figure(figsize=(w, h))
    for i in range(num_plots):
        # ax = fig.add_subplot(num_plots, 1, i+1, frameon=False)
        ax = fig.add_subplot(num_plots, 1, i+1, frameon=True)
        axplot_multi_lines(ax=ax, data=data_plots[i])    
    fig.tight_layout()
    fig.savefig(os.path.join(root,file), dpi=200)
    plt.close()

    
def axplot_multi_lines(ax, data):
    """
        Docs: plot a figure with data and saved in file
        Args:
            file: pdf prefered
            data: dic with keys including 'x', 'y', 'color', 'title', 'legend'
            
            e.g.: data = [
                        {title':'', 'x_label':'', 'y_label':'', 'y_axis':[0,1],},
                        {'x':, 'y':, 'color':'blue', 'label':'nat.', 'linewidth':0.5},
                        {'x':, 'y':, 'color':'tomato', 'label':'adv.', 'linewidth':0.5}
                    ]
    """
    assert isinstance(data, list)
    for i in range(1,len(data)):
        line = data[i]
        x = line['x']
        y = line['y']
        color = line['color']
        label = line['label']
        linewidth=line['linewidth']
        ax.plot(x, y, '-', color=color, alpha=0.8/math.sqrt(i+1), linewidth=linewidth, label=label)
        # ax.fill_between(x, y, np.zeros_like(x), facecolor=color, alpha=0.8/math.sqrt(i+1), label=label)

    ax.legend(loc='upper center', markerscale=1, fancybox=True, fontsize=12)

    title = data[0]['title']
    x_label = data[0]['x_label']
    y_label = data[0]['y_label']
    y_axis = data[0]['y_axis']
    ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if y_axis is not None:
        ax.set_ylim(min(y_axis[0]), max(y_axis[1]))

    ax.margins(x=0.01)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set(facecolor = "ivory")
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=.5)
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=.5)
    
    


if __name__ == "__main__":
    pass

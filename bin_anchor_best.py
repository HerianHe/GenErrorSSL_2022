# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

"""
    TO-DOs:
    - in each loop, define train/sub-unlabled-set, train for a loop to convergence
"""
# %%
import os, sys, argparse, pathlib, itertools, tqdm, random, copy
import numpy as np
from utils import timer, log_test_train_acc_loss_history
        
parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--SEED', default=0, type=int)
parser.add_argument('--GPU_IDs', nargs='+', default=[0], type=int)
parser.add_argument('--is_Train', action='store_true')
parser.add_argument('--root', default='./experiments/icml')

parser.add_argument('--exp_name', default='CIFAR10')
parser.add_argument('--network', default='RGB10')
parser.add_argument('--classes', nargs='+', default=[1,7], type=int)
parser.add_argument('--num_labeled', default=5000, type=int)
parser.add_argument('--ratio_unlabeled', default=10, type=int)

parser.add_argument('--test_batch_size', default=100, type=int)

parser.add_argument('--tr_loops', default=3, type=int)
parser.add_argument('--init_epochs', default=100, type=int)
parser.add_argument('--init_batch_size', default=10, type=int)
parser.add_argument('--tr_epochs', default=2, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--ratio_loop', default=1, type=int)
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--milestones_ratio', default=[0.25, 0.8], type=list)

parser.add_argument('--version', default=0, type=int)

parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint', type=str, default='experiments/./nets/loop_0_ckp_best.pt')
parser.add_argument('--net_only', default=True, type=lambda x: bool(int(x)))

args, unknown = parser.parse_known_args()


# %%
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import datasets, transforms
from torchvision.utils import save_image
from ignite.metrics import Accuracy, Average
# if args.is_Train:
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)
os.environ['PYTHONHASHSEED'] = str(args.SEED) # to avoid hash random to make the experiment repeatable
torch.cuda.manual_seed_all(args.SEED) # set random seed to all GPUs
torch.backends.cudnn.deterministic = True


EXP_PATH = os.path.join(args.root, 
    args.exp_name + f'_lb{args.num_labeled}_m{args.ratio_unlabeled}_lp{args.tr_loops}_ep{args.tr_epochs}_'+\
        f'lr{args.lr}_wd{args.weight_decay}_sd{args.SEED}_rl{args.ratio_loop}_bs{args.train_batch_size}_v{args.version}')
pathlib.Path(os.path.join(EXP_PATH, 'nets')).mkdir(parents=True, exist_ok=True)




# %%
# Networks
sys.path.append('..')

if args.network == 'RGB10':
    from models.nets.resnet import ResNet_RGB10 as classification_network
elif args.network == 'RGB8':
    from models.nets.resnet import ResNet_RGB8 as classification_network
# %%
# Model Object

class Classifier():
    def __init__(self, args=None, device='cuda', is_train=False) -> None:
        super().__init__()
        self.net = classification_network(num_classes=2).to(device)
        self.is_train = is_train
        self.device = device
        self.GPU_IDs = args.GPU_IDs
        self.metric_accuracy = Accuracy()
        self.metric_loss = Average()
        
        if len(self.GPU_IDs) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=self.GPU_IDs)
        if is_train:
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            # self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            self.scheduler = None
            self.criterion = nn.CrossEntropyLoss()
            self.tr_loops = args.tr_loops
            self.init_epochs = args.init_epochs
            self.init_batch_size = args.init_batch_size
            self.tr_epochs = args.tr_epochs
            self.test_batch_size = args.test_batch_size
            self.train_batch_size = args.train_batch_size
            self.ratio_loop = args.ratio_loop
            self.log_interval = 5
            self.net_best = None
        else:
            self.eval_mode()
            self.set_requires_grad([self.net], False)
    
    def eval_mode(self):
        self.net.eval()
    def train_mode(self):
        self.net.train()
    def load_networks(self, path):
        self.checkpoint = torch.load(path)
        if len(self.GPU_IDs) == 1:
            self.net.load_state_dict(self.checkpoint['state_dict'])
        else:
            self.net.module.load_state_dict(self.checkpoint['state_dict'])

    
    def fit(self, labeled_trainset, unlabeled_trainset, testset, logger, writer=None, save_path='.'):
        assert len(labeled_trainset) <= len(unlabeled_trainset)
        test_loader = data.DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=4)
        hist_loops = log_test_train_acc_loss_history()
        for loop in range(0, self.tr_loops):
            if loop == 0:
                labeled_train_loader = data.DataLoader(labeled_trainset, batch_size=self.init_batch_size, shuffle=True, num_workers=4)
                unlabeled_train_loader = None
                test_acc, test_loss, train_acc, train_loss = self._fit_one_loop(labeled_train_loader, unlabeled_train_loader, test_loader, loop, self.init_epochs, logger, os.path.join(save_path, 'nets'))
            else:
                bs_label = self.train_batch_size // 2
                bs_unlabel = bs_label
                labeled_train_loader = data.DataLoader(labeled_trainset, batch_size=bs_label, shuffle=True, num_workers=4)
                indices = torch.randperm(len(unlabeled_trainset))
                unlabeled_trainset_subset = data.Subset(unlabeled_trainset, indices[:len(labeled_trainset)*self.ratio_loop])
                unlabeled_train_loader = data.DataLoader(unlabeled_trainset_subset, batch_size=bs_unlabel, shuffle=True, num_workers=4)
                if loop < 3:
                    tr_epochs = int(self.init_epochs * (0.5**loop))
                else:
                    tr_epochs = int(self.tr_epochs)
                test_acc, test_loss, train_acc, train_loss = self._fit_one_loop(labeled_train_loader, unlabeled_train_loader, test_loader, loop, tr_epochs, logger, os.path.join(save_path, 'nets'))
                # test_acc, test_loss, train_acc, train_loss = self._fit_one_loop(labeled_train_loader, unlabeled_train_loader, test_loader, loop, self.tr_epochs, logger, os.path.join(save_path, 'nets'))
                
            
            logger.info(f'****************** [Loop {loop}], test_acc {test_acc:.5f}, test_loss {test_loss:.5f}, train_acc {train_acc:.5f}, train_loss {train_loss:.5f} ******************\n\n')
            hist_loops.append_results([loop, test_acc, test_loss, train_acc, train_loss])
            hist_loops.visualize(os.path.join(save_path, 'vis.png'))
            hist_loops.save_results(save_path)
            
    def _fit_one_loop(self, labeled_train_loader, unlabeled_train_loader, test_loader, loop, tr_epochs, logger, save_path):        
        assert (labeled_train_loader is not None) or (unlabeled_train_loader is not None)
        logger.info(f'***** Training Loop: {loop} *****')
        tic_toc = timer()
        hist_epochs = log_test_train_acc_loss_history()
        # unlabeled_data_labeling_func = None if unlabeled_train_loader is None else (unlabeled_train_loader, copy.deepcopy(self.net))
        unlabeled_data_labeling_func = None if unlabeled_train_loader is None else (unlabeled_train_loader, copy.deepcopy(self.net_best))
        for epoch in range(tr_epochs):
            self._fit_one_epoch(labeled_train_loader, unlabeled_data_labeling_func, epoch, logger=logger)
            test_acc, test_loss = self._evaluate(epoch, test_loader, None)
            train_acc, train_loss = self._evaluate(epoch, labeled_train_loader, unlabeled_data_labeling_func)
            hist_epochs.append_results([epoch, test_acc, test_loss, train_acc, train_loss])
            model_state_dict = self.net.state_dict() if len(self.GPU_IDs)<2 else self.net.module.state_dict()
            # checkpoint = {'state_dict':model_state_dict, 'stop_epoch':epoch, 'optimizer': self.optimizer.state_dict()}
            best_epoch = hist_epochs.best_timepoint
            if best_epoch == epoch:
                # torch.save(checkpoint, os.path.join(save_path, 'loop_'+str(loop)+"_ckp_best.pt"))
                self.net_best = copy.deepcopy(self.net)
            logger.info(f'Current result {test_acc} @ {epoch :03d}, '
                        f'Time for Epoch:{tic_toc.toc():.03f}; '
                        f'Best result {hist_epochs.best_test_acc} @ {best_epoch :03d} \n')
        # return test_acc, test_loss, train_acc, train_loss
        # return hist_epochs.best_test_acc, hist_epochs.best_test_loss, train_acc, train_loss
        return hist_epochs.best_test_acc, hist_epochs.best_test_loss, hist_epochs.best_train_acc, hist_epochs.best_train_loss,
    
    def _fit_one_epoch(self, labeled_train_loader, unlabeled_data_labeling_func, epoch, logger):
        logger.info('Training Epoch: {}, Current learning rate:{:.6f}'.format(epoch, self.optimizer.param_groups[0]['lr']) )
        tic_toc = timer()
        if unlabeled_data_labeling_func is None:
            for batch_idx, (data_lb, targets_lb) in enumerate(labeled_train_loader):
                data = data_lb.to(self.device)
                targets = targets_lb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(data), targets)
                loss.backward()
                self.optimizer.step()
                if (batch_idx) % self.log_interval == 0:
                    logger.info(f'[{batch_idx*len(data)}/{len(labeled_train_loader)*len(data)} ({100. * batch_idx / len(labeled_train_loader) :.0f}%)], '
                                f'Loss: {loss.item() :.6f}, Time for Batches: {tic_toc.toc() :03f}')
        else:
            unlabeled_train_loader, labeling_func = unlabeled_data_labeling_func
            labeled_train_iter = iter(labeled_train_loader)
            for batch_idx, (data_ulb, _) in enumerate(unlabeled_train_loader):
                data_ulb = data_ulb.to(self.device)
                with torch.no_grad():
                    targets_pesudo = torch.argmax(labeling_func(data_ulb), dim=1)
                try:
                    data_lb, targets_lb = labeled_train_iter.next()
                except:
                    labeled_train_iter = iter(labeled_train_loader)
                    data_lb, targets_lb = labeled_train_iter.next()
                data_lb, targets_lb = data_lb.to(self.device), targets_lb.to(self.device)
                data = torch.cat((data_lb, data_ulb), dim=0)
                targets = torch.cat((targets_lb, targets_pesudo), dim=0)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(data), targets)
                loss.backward()
                self.optimizer.step()
                if (batch_idx) % self.log_interval == 0:
                    logger.info(f'[{batch_idx*len(data)}/{len(labeled_train_loader)*len(data)} ({100. * batch_idx / len(labeled_train_loader) :.0f}%)], '
                                f'Loss: {loss.item() :.6f}, Time for Batches: {tic_toc.toc() :03f}')
                
        if self.scheduler is not None:
            self.scheduler.step()

    def _evaluate(self, epoch, labeled_data, unlabeled_data_labeling_func, writer=None, tag='test'):
        self.eval_mode()
        self.metric_accuracy.reset()
        self.metric_loss.reset()
        
        if labeled_data is not None:
            print('evaluting on labled data...')
            for data, targets in labeled_data:
                data = data.to('cuda')
                targets = targets.to('cuda')
                with torch.no_grad():
                    predictions = self.net(data)
                    loss = F.cross_entropy(predictions, targets, reduction='none')
                self.metric_accuracy.update((predictions, targets))
                self.metric_loss.update(loss.unsqueeze(dim=1))
            
        if unlabeled_data_labeling_func is not None:
            print('evaluting on unlabled data...')
            unlabeled_loader, labeling_func = unlabeled_data_labeling_func
            for data, _ in unlabeled_loader:
                data = data.to('cuda')
                with torch.no_grad():
                    targets_pesudo = torch.argmax(labeling_func(data), dim=1)
                    predictions = self.net(data)
                    loss = F.cross_entropy(predictions, targets_pesudo, reduction='none')
                self.metric_accuracy.update((predictions, targets_pesudo))
                self.metric_loss.update(loss.unsqueeze(dim=1))
                
        accuracy = self.metric_accuracy.compute()
        loss_avg = self.metric_loss.compute()
        return accuracy, loss_avg.numpy()[0]


# %%
# if __name__ == '__main__':
#     if args.is_Train:
from utils import get_logger
logger = get_logger(os.path.join(EXP_PATH, 'logging.txt'))
logger.info(args)
model = Classifier(args, is_train=True)
logger.info(model.net)

# %%
DATA_PATH = './Data'
pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

Dict_Classes = dict([
                    (0, 'airplane'),
                    (1, 'automobile'),
                    (2, 'bird'),
                    (3, 'cat'),
                    (4, 'deer'),
                    (5, 'dog'),
                    (6, 'frog'),
                    (7, 'horse'),
                    (8, 'ship'),
                    (9, 'truck')
                    ])
logger.info(f'Classes of interest are {Dict_Classes[args.classes[0]]} and {Dict_Classes[args.classes[1]]}')
def get_indices_by_class(dataset, class_list):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_list:
            indices.append(i)
    return indices

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

class target_transform():
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, y):
        assert y in self.classes
        return self.classes.index(y)

trainset = datasets.CIFAR10(DATA_PATH, train=True, download=True, transform=transform_train, target_transform=target_transform(args.classes))
trainset = data.Subset(trainset, get_indices_by_class(trainset, args.classes))

indices = torch.randperm(len(trainset))
N = args.num_labeled
M = N * args.ratio_unlabeled if N * (1+args.ratio_unlabeled)<=len(indices) else len(indices)-N
labeled_trainset = data.Subset(trainset, indices[:N])
unlabeled_trainset = data.Subset(trainset, indices[N:N+M])

testset = datasets.CIFAR10(DATA_PATH, train=False, download=True, transform=transform_test, target_transform=target_transform(args.classes))
testset = data.Subset(testset, get_indices_by_class(testset, args.classes))


model.fit(labeled_trainset=labeled_trainset, unlabeled_trainset=unlabeled_trainset, testset=testset,
    logger=logger, writer=None, save_path=EXP_PATH)
    
    

# %%
"""
# Dataset
DATA_PATH = './Data'
pathlib.Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

trainset = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
indices = torch.randperm(len(trainset))
N = args.num_labeled
M = N * args.ratio_unlabeled if N * (1+args.ratio_unlabeled)<=len(indices) else len(indices)-N
labeled_trainset = data.Subset(trainset, indices[:N])
unlabeled_trainset = data.Subset(trainset, indices[N:N+M])

labeled_train_loader = data.DataLoader(labeled_trainset, batch_size=args.train_batch_size, num_workers=4)
unlabeled_train_loader = data.DataLoader(unlabeled_trainset, batch_size=args.train_batch_size * args.ratio_unlabeled, num_workers=4)
test_loader = data.DataLoader(datasets.MNIST(DATA_PATH, train=False, transform=transforms.ToTensor()), batch_size=args.test_batch_size, shuffle=False, num_workers=4)
print(labeled_train_loader.dataset.__len__())


# if True:
if args.labeled_only:
    model.fit(labeled_train_loader=labeled_train_loader, unlabeled_train_loader=None, test_loader=test_loader,
        logger=logger, writer=None, save_path=os.path.join(EXP_PATH, 'nets'))
else:
    model.load_networks(args.checkpoint)
    model.fit(labeled_train_loader=labeled_train_loader, unlabeled_train_loader=unlabeled_train_loader, test_loader=test_loader,
        logger=logger, writer=None, save_path=os.path.join(EXP_PATH, 'nets'))
"""
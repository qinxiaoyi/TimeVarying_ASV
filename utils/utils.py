#! /usr/bin/env python3
import torch, os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_ramdom_state(chk_dir, ramdom_state, np_stats, torch_state, torch_cuda_state):
    torch.save({'random': ramdom_state,
                'np': np_stats,
                'torch': torch_state,
                'torch_cuda': torch_cuda_state
               }, os.path.join(chk_dir, 'random_state.pkl'))
    
def save_checkpoint(chk_dir, epoch, model, classifier, optimizer, scheduler=None, lr=None):
    torch.save({'model': model.module.state_dict(),
                'classifier': classifier.state_dict() if classifier else None,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'lr': lr
               }, os.path.join(chk_dir, 'model_%d.pkl' % epoch))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn

def sliding_window(lst, window_size, stride=1):
    """
    将列表 lst 按照给定的窗口大小和步长进行滑动窗口切分。
    参数：
        - lst：要进行切分的列表。
        - window_size：窗口大小，即每个子列表的长度。
        - stride：步长，即每次滑动的距离，默认为 1。
    返回值：
        - 一个由子列表组成的列表，每个子列表的长度为 window_size。
    """
    if window_size > len(lst):
        raise ValueError("Window size cannot be larger than list length")
    if stride > window_size:
        raise ValueError("Stride cannot be larger than window size")
    num_windows = (len(lst) - window_size) // stride + 1
    windows = []
    for i in range(num_windows):
        windows.append(lst[i*stride:i*stride+window_size])
    return windows

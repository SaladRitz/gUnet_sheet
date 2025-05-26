!pip install pytorch==1.12 torchvision torchaudio cudatoolkit==11.3
!pip install timm==0.6.5
!pip install pytorch-msssim
!pip install opencv-python==4.5.5.62
!pip install tqdm tensorboard tensorboardx

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

#from utils import AverageMeter, CosineScheduler, pad_img
#from datasets import PairLoader
#from models import *

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import subprocess
import time
import numpy as np
import shutil






#--------------------------------------------------UTILS FOLDER--------------------------------------------------------------------

import math
import torch

from timm.scheduler.scheduler import Scheduler


class CosineScheduler(Scheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_name: str,
                 t_max: int,
                 value_min: float = 0.,
                 warmup_t=0,
                 const_t=0,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field=param_name, initialize=initialize)

        assert t_max > 0
        assert value_min >= 0
        assert warmup_t >= 0
        assert const_t >= 0
        
        #self.t_max = t_max 

        self.cosine_t = t_max - warmup_t - const_t
        self.value_min = value_min
        self.warmup_t = warmup_t
        self.const_t = const_t

        if self.warmup_t:
            self.warmup_steps = [(v - value_min) / self.warmup_t for v in self.base_values]
            super().update_groups(self.value_min)
        else:
            self.warmup_steps = []

    def _get_value(self, t):
        if t < self.warmup_t:
            values = [self.value_min + t * s for s in self.warmup_steps]
        elif t < self.warmup_t + self.const_t:
            values = self.base_values
        else:
            t = t - self.warmup_t - self.const_t

            value_max_values = [v for v in self.base_values]

            values = [
                self.value_min + 0.5 * (value_max - self.value_min) * (1 + math.cos(math.pi * t / self.cosine_t))
                for value_max in value_max_values
            ]

        return values

    def get_epoch_values(self, epoch: int):
        return self._get_value(epoch)


    def _get_lr(self):
        #Compute the learning rate for the current epoch.
        if self.last_epoch < self.warmup_t:
            # Linear warmup phase
            lr = self.value_min + (self.base_lrs[0] - self.value_min) * (self.last_epoch / self.warmup_t)
        elif self.last_epoch < self.warmup_t + self.const_t:
            # Constant learning rate phase
            lr = self.base_lrs[0]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_t - self.const_t) / (self.t_max - self.warmup_t - self.const_t)
        lr = self.value_min + 0.5 * (self.base_lrs[0] - self.value_min) * (1 + torch.cos(progress * 3.141592653589793))
    
        return [lr for _ in self.optimizer.param_groups]




class AverageMeter(object):
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


def pad_img(x, patch_size):
	_, _, h, w = x.size()
	mod_pad_h = (patch_size - h % patch_size) % patch_size
	mod_pad_w = (patch_size - w % patch_size) % patch_size
	x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
	return x







#------------------------------------------------------DATASETS FOLDER---------------------------------------------------------------
from torch.utils.data import Dataset
import cv2
import random

def read_img(filename, to_float=True):
	img = cv2.imread(filename)
	if to_float: img = img.astype('float32') / 255.0
	return img[:, :, ::-1]


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	if data_augment:
		# horizontal flip
		if random.randint(0, 1) == 1:
			for i in range(len(imgs)):
				imgs[i] = np.flip(imgs[i], axis=1)

		# bad data augmentations for outdoor dehazing
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs





class PairLoader(Dataset):
	def __init__(self, root_dir, mode, size=256, edge_decay=0, data_augment=True, cache_memory=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.data_augment = data_augment

		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

		self.cache_memory = cache_memory
		self.source_files = {}
		self.target_files = {}

	def __len__(self):
		return self.img_num
	
	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)
		img_name = self.img_names[idx]

    	# Try to load from cache if cache_memory is enabled
		if self.cache_memory and img_name in self.source_files:
			source_img = self.source_files[img_name]
			target_img = self.target_files[img_name]
		else:
        	# Fallback: Load from disk if not cached
			source_img = read_img(os.path.join(self.root_dir, 'IN', img_name), to_float=False)
			target_img = read_img(os.path.join(self.root_dir, 'GT', img_name), to_float=False)

        	# Cache the images in memory if cache_memory is enabled
			if self.cache_memory:
				self.source_files[img_name] = source_img
				self.target_files[img_name] = target_img

    	# Convert to [-1, 1] range
		source_img = source_img.astype('float32') / 255.0 * 2 - 1
		target_img = target_img.astype('float32') / 255.0 * 2 - 1

    	# Apply data augmentation or alignment
		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)
		elif self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

    	# Convert to PyTorch tensors
		source_tensor = torch.tensor(hwc_to_chw(source_img), dtype=torch.float32)
		target_tensor = torch.tensor(hwc_to_chw(target_img), dtype=torch.float32)
		return {'source': source_tensor, 'target': target_tensor, 'filename': img_name}

#----------------------------------------------------------------------------------------------------------------------------------



#                                    ------------GUNET.PY FOLDER-------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
#from .norm_layer import *


class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect')
		)

		self.Wg = nn.Sequential(
			nn.Conv2d(dim, dim, 1),
			gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=True)
		)

		self.proj = nn.Conv2d(dim, dim, 1)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		out = self.Wv(X) * self.Wg(X)
		out = self.proj(out)
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


class gUNet(nn.Module):
	def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(gUNet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size, 
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)

		feat = self.layers[self.half_num](feat)

		for i in range(self.half_num-1, -1, -1):
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		x = self.outconv(feat) + x

		return x


__all__ = ['gUNet', 'gunet_t', 'gunet_s', 'gunet_b', 'gunet_d']

# Normalization batch size of 16~32 may be good

def gunet_t():	# 4 cards 2080Ti
	return gUNet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_s():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_b():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def gunet_d():	# 4 cards 3090
	return gUNet(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)






def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d
       to SynchronizedBatchNorm*N*d

    Args:
        module: the input module needs to be convert to SyncBN model

    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using SyncBN
        >>> m = convert_model(m)
    """
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod.cuda()
        mod = DataParallelWithCallback(mod, device_ids=module.device_ids)
        return mod

    mod = module
    for pth_module, sync_module in zip([torch.nn.BatchNorm1d,
                                        torch.nn.BatchNorm2d,
                                        torch.nn.BatchNorm3d],
                                       [SynchronizedBatchNorm1d,
                                        SynchronizedBatchNorm2d,
                                        SynchronizedBatchNorm3d]):
        if isinstance(module, pth_module):
            mod = sync_module(module.num_features, module.eps, module.momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod


class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.

    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules
	

def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)

class CallbackContext(object):
    pass



#                                               BATCHNORM.PY IN SYNC_BATCHNORM FOLDER

import collections
import contextlib

import torch
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast
except ImportError:
    ReduceAddCoalesced = Broadcast = None



__all__ = [
    'set_sbn_eps_mode',
    'SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d',
    'patch_sync_batchnorm', 'convert_model'
]


SBN_EPS_MODE = 'clamp'


def set_sbn_eps_mode(mode):
    global SBN_EPS_MODE
    assert mode in ('clamp', 'plus')
    SBN_EPS_MODE = mode


def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dimensions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])







class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        assert ReduceAddCoalesced is not None, 'Can not use Synchronized Batch Normalization without CUDA support.'

        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                                                     track_running_stats=track_running_stats)

        if not self.track_running_stats:
            import warnings
            warnings.warn('track_running_stats=False is not supported by the SynchronizedBatchNorm.')

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        assert input.size(1) == self.num_features, 'Channel size mismatch: got {}, expect {}.'.format(input.size(1), self.num_features)
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i*2:i*2+2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        if hasattr(torch, 'no_grad'):
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        if SBN_EPS_MODE == 'clamp':
            return mean, bias_var.clamp(self.eps) ** -0.5
        elif SBN_EPS_MODE == 'plus':
            return mean, (bias_var + self.eps) ** -0.5
        else:
            raise ValueError('Unknown EPS mode: {}.'.format(SBN_EPS_MODE))




class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
	
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))



#                                                COMM.PY IN SYNC_BATCHNORM FOLDER


import queue
import collections
import threading

__all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.

    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """

        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.

        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).

        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.

        Returns: the message to be sent back to the master device.

        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)

#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------




#GPU Config
# Automatically use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gunet_t', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--use_mp', action='store_true', default=False, help='use Mixed Precision')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
#parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--save_dir', default='/kaggle/working/saved_models/', type=str, help='path to models saving')
#parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')   # Keep the base data path
parser.add_argument('--data_dir', default='/kaggle/input/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')

# Update dataset names to match your folder structure
parser.add_argument('--train_set', default='haze4k-t/Haze4K-T', type=str, help='train dataset name')   # Train dataset
parser.add_argument('--val_set', default='haze4k-v/Haze4K-V', type=str, help='valid dataset name')     # Validation dataset

# Experiment settings
parser.add_argument('--exp', default='reside-in', type=str, help='experiment setting')

#args = parser.parse_args()
args, _ = parser.parse_known_args()

# training environment
if args.use_ddp:
	torch.distributed.init_process_group(backend='nccl', init_method='env://')
	world_size = dist.get_world_size()
	local_rank = dist.get_rank()
	torch.cuda.set_device(local_rank)
	if local_rank == 0: print('==> Using DDP.')
else:
	world_size = 1


# training config

b_setup = {
    "t_patch_size": 256,             
    "valid_mode": "valid",
    "v_patch_size": 400,             
    "v_batch_ratio": 1.0,
    "edge_decay": 0.1,
    "weight_decay": 0.01,
    "data_augment": True,
    "cache_memory": False,
    "num_iter": 16384,                
    "epochs": 1000,                   
    "warmup_epochs": 50,             
    "const_epochs": 0,
    "frozen_epochs": 200,
    "eval_freq": 10                    
}

variant = args.model.split('_')[-1]
config_name = 'model_'+variant+'.json' if variant in ['t', 's', 'b', 'd'] else 'default.json'	# default.json as baselines' configuration file 

m_setup = {
"batch_size": 2,
"lr": 8e-4
}


def reduce_mean(tensor, nprocs):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= nprocs
	return rt


def train(train_loader, network, criterion, optimizer, scaler, epochs, frozen_bn=False, plot_interval=50):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.eval() if frozen_bn else network.train()	# simplified implementation that other modules may be affected
	

#	for batch in enumerate(train_loader):
#		source_img = batch['source'].cuda()
#		target_img = batch['target'].cuda()
	batch_count = 0  # ✅ Counter to track batch index
	for batch in train_loader:
		batch_count += 1  # ✅ Increment batch counter manually

		# ✅ Access images from dictionary
		source_img = batch['source'].to('cuda')
		target_img = batch['target'].to('cuda')
		output = network(source_img)


		with autocast(args.use_mp):
			output = network(source_img)
			loss = criterion(output, target_img)

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		if args.use_ddp: loss = reduce_mean(loss, dist.get_world_size())
		losses.update(loss.item())

#		# Show images during training
#		if batch % 10 == 0:
#			show_images(source_img, target_img, output, epochs, batch)

        # ✅ Automatically plot and save images every `plot_interval` batches
		if batch_count % plot_interval == 0:
			#print(f"\n🛠️ Plotting images at Epoch {epochs}, Batch {batch_count}...")
			show_images(source_img, target_img, output, epochs, batch_count)



	return losses.avg



# New Validation loss function

from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
'''
def valid(val_loader, network, criterion):
    """Returns (avg_psnr, avg_ssim, avg_val_loss) over val_loader."""
    psnr_meter = AverageMeter()
    loss_meter = AverageMeter()
    ssim_meter = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        src = batch['source'].to('cuda')
        tgt = batch['target'].to('cuda')

        with torch.no_grad():
            H, W = src.shape[2:]
            pad = (network.module.patch_size
                   if hasattr(network.module, 'patch_size')
                   else 16)
            src_p = pad_img(src, pad)
            out = network(src_p).clamp_(-1, 1)
            out = out[:, :, :H, :W]

        # compute loss
        loss = criterion(out, tgt).item()
        loss_meter.update(loss, src.size(0))

        # compute PSNR
        mse_per_image = F.mse_loss(out * 0.5 + 0.5,
                                   tgt * 0.5 + 0.5,
                                   reduction='none') \
                         .mean((1,2,3))
        psnr_batch = (10 * torch.log10(1 / mse_per_image)).mean().item()
        psnr_meter.update(psnr_batch, src.size(0))

        # compute SSIM (per-image, on CPU numpy)
        out_np = (out * 0.5 + 0.5).cpu().numpy()
        tgt_np = (tgt * 0.5 + 0.5).cpu().numpy()
        # loop over batch
        ssim_sum = 0
        for i in range(out_np.shape[0]):
            # assume channels-first RGB or grayscale
            # for multichannel=True if C>1
            s = compare_ssim(out_np[i].transpose(1,2,0),
                             tgt_np[i].transpose(1,2,0),
                             data_range=1.0,
                             multichannel=True)
            ssim_sum += s
        ssim_batch = ssim_sum / out_np.shape[0]
        ssim_meter.update(ssim_batch, src.size(0))

    return psnr_meter.avg, ssim_meter.avg, loss_meter.avg

'''
def valid(val_loader, network, criterion):
    """Returns (avg_psnr, avg_ssim, avg_val_loss) over val_loader."""
    psnr_meter = AverageMeter()
    loss_meter = AverageMeter()
    ssim_meter = AverageMeter()

    torch.cuda.empty_cache()
    network.eval()

    for batch in val_loader:
        src = batch['source'].to('cuda')
        tgt = batch['target'].to('cuda')

        with torch.no_grad():
            H, W = src.shape[2:]
            pad = (network.module.patch_size
                   if hasattr(network.module, 'patch_size')
                   else 16)
            src_p = pad_img(src, pad)
            out = network(src_p).clamp_(-1, 1)
            out = out[:, :, :H, :W]

        # 1) compute and record L1 loss
        val_loss = criterion(out, tgt).item()
        loss_meter.update(val_loss, src.size(0))

        # 2) compute and record PSNR
        mse_per_image = F.mse_loss(out * 0.5 + 0.5,
                                   tgt * 0.5 + 0.5,
                                   reduction='none') \
                         .mean((1, 2, 3))
        psnr_batch = (10 * torch.log10(1 / mse_per_image)).mean().item()
        psnr_meter.update(psnr_batch, src.size(0))

        # 3) compute and record SSIM per image
        out_np = (out * 0.5 + 0.5).cpu().numpy()  # shape: (N, C, H, W)
        tgt_np = (tgt * 0.5 + 0.5).cpu().numpy()
        batch_ssim = 0.0

        for i in range(out_np.shape[0]):
            img = out_np[i]
            ref = tgt_np[i]
            C, h, w = img.shape

            # pick an odd window size between 3 and min(h, w)
            win_size = min(h, w)
            if win_size % 2 == 0:
                win_size -= 1
            win_size = max(win_size, 3)

            # prepare HxW or HxWxC array
            img_hw = img.transpose(1, 2, 0)
            ref_hw = ref.transpose(1, 2, 0)

            if C == 1:
                # squeeze singleton channel for grayscale
                img_hw = img_hw[:, :, 0]
                ref_hw = ref_hw[:, :, 0]
                channel_axis = None
            else:
                channel_axis = -1

            s = compare_ssim(
                img_hw,
                ref_hw,
                data_range=1.0,
                win_size=win_size,
                channel_axis=channel_axis
            )
            batch_ssim += s

        ssim_batch = batch_ssim / out_np.shape[0]
        ssim_meter.update(ssim_batch, src.size(0))

    return psnr_meter.avg, ssim_meter.avg, loss_meter.avg







# Visualization Function


def show_images(orig, target, output, epoch, batch, save_dir="./plots", display_time=3):
    """ 
    Visualize and save original, target, and output images side by side.
    Automatically close and save the images.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    plt.ion()  # ✅ Enable interactive mode
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Convert tensors to NumPy and transpose for display
    orig = orig.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    target = target.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
    output = output.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]

    # Normalize and clip the values to [0,1] range
    orig = (orig * 0.5 + 0.5).clip(0, 1)
    target = (target * 0.5 + 0.5).clip(0, 1)
    output = (output * 0.5 + 0.5).clip(0, 1)

    # Display images
    axes[0].imshow(orig)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(target)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(output)
    axes[2].set_title('Output')
    axes[2].axis('off')

    plt.suptitle(f'Epoch: {epoch}, Batch: {batch}')

    # ✅ Save the image
    plot_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch}.png")
    plt.savefig(plot_path)
    print(f"✅ Saved plot: {plot_path}")

    # ✅ Display for `display_time` seconds, then close automatically
    plt.pause(display_time)
    plt.close(fig)  # ✅ Automatically close the plot
    plt.ioff()  # Disable interactive mode





def find_latest_checkpoint(model_name, save_dir):
	files = [f for f in os.listdir(save_dir) if f.startswith(model_name) and 'epoch' in f and f.endswith('.pth')]
	if not files:
		return None
	files.sort(key=lambda f: int(re.findall(r'epoch(\d+)', f)[0]))
	return os.path.join(save_dir, files[-1])

def resume_or_initialize(model, optimizer, lr_scheduler, wd_scheduler, scaler, save_dir, model_name):
	checkpoint_path = find_latest_checkpoint(model_name, save_dir)
	if checkpoint_path is None:
		print("🆕 No checkpoint found. Starting fresh.")
		history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []}
		return 0, 0, history
	else:
		print(f"🔁 Resuming from checkpoint: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
		wd_scheduler.load_state_dict(checkpoint['wd_scheduler'])
		scaler.load_state_dict(checkpoint['scaler'])
		history = checkpoint.get('history', {'epoch': [], 'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []})
		return checkpoint['cur_epoch'], checkpoint['best_psnr'], history












def main():
    import os
    import glob
    import zipfile
    import shutil
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from torch.utils.data import DataLoader, RandomSampler
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp import GradScaler
    from tqdm import tqdm

    # 1) Set up save directory under Kaggle working
    save_dir = '/kaggle/working/saved_models'
    os.makedirs(save_dir, exist_ok=True)

    # 2) Unzip any uploaded checkpoint zips into save_dir
    for z in glob.glob('/kaggle/input/*.zip'):
        with zipfile.ZipFile(z, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    # Also copy .pth checkpoints from all datasets (not zipped)
    for root, dirs, files in os.walk('/kaggle/input/'):
        for f in files:
            if f.endswith('.pth'):
                full_path = os.path.join(root, f)
                print("Found checkpoint:", full_path)
                shutil.copy(full_path, save_dir)

    # 3) History for CSV
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'psnr': [], 'ssim': []}

    # 4) Build model
    network = eval(args.model)().cuda()
    if args.use_ddp:
        network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
        if m_setup['batch_size'] // world_size < 16:
            if local_rank == 0:
                print('==> Using SyncBN (small batch per GPU)')
            nn.SyncBatchNorm.convert_sync_batchnorm(network)
    else:
        network = DataParallel(network)
        if m_setup['batch_size'] // torch.cuda.device_count() < 16:
            print('==> Using SyncBN (small batch overall)')
            convert_model(network)

    # 5) Loss, optimizer, schedulers, scaler
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'],
                                  weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr',
                                   t_max=b_setup['epochs'],
                                   value_min=m_setup['lr'] * 1e-2,
                                   warmup_t=b_setup['warmup_epochs'],
                                   const_t=b_setup['const_epochs'])
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay',
                                   t_max=b_setup['epochs'])
    scaler = GradScaler()

    # 6) Resume logic: find latest checkpoint

    def latest_ckpt(save_dir, model_name):
        snaps = sorted(
            glob.glob(os.path.join(save_dir, f"{model_name}_epoch*.pth")),
            key=lambda f: int(os.path.splitext(f)[0].split('epoch')[-1])
        )
        return snaps[-1] if snaps else None

    ckpt = latest_ckpt(save_dir, args.model)
    if ckpt:
        print(f"==> Resuming from checkpoint {ckpt}")
        #m = torch.load(ckpt, map_location='cpu')
        m = torch.load(ckpt, map_location='cpu', weights_only=False)

        network.load_state_dict(m['state_dict'])
        optimizer.load_state_dict(m['optimizer'])
        lr_scheduler.load_state_dict(m['lr_scheduler'])
        wd_scheduler.load_state_dict(m['wd_scheduler'])
        scaler.load_state_dict(m['scaler'])
        cur_epoch = m.get('cur_epoch', 0)
        best_psnr = m.get('best_psnr', 0)
        history = m.get('history', history)
    else:
        print("==> No checkpoint found, starting fresh")
        cur_epoch = 0
        best_psnr = 0
    # 7) Data loaders
    train_dataset = PairLoader(os.path.join(args.data_dir, args.train_set), 'train',
                               b_setup['t_patch_size'], b_setup['edge_decay'],
                               b_setup['data_augment'], b_setup['cache_memory'])
    train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              sampler=RandomSampler(train_dataset,
                                                    num_samples=b_setup['num_iter']//world_size),
                              num_workers=4, pin_memory=True,
                              drop_last=True, persistent_workers=True)

    val_dataset = PairLoader(os.path.join(args.data_dir, args.val_set),
                             b_setup['valid_mode'], b_setup['v_patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=max(int(m_setup['batch_size'] *
                                               b_setup['v_batch_ratio']//world_size), 1),
                            num_workers=args.num_workers//world_size,
                            pin_memory=True)

    # 8) TensorBoard
    if not args.use_ddp or local_rank == 0:
        print('==> Start training:', args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    # 9) Training loop
    for epoch in tqdm(range(cur_epoch, b_setup['epochs']+1)):
        frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])
        # **Train**
        loss = train(train_loader, network, criterion, optimizer, scaler, epoch, frozen_bn)
        lr_scheduler.step(epoch+1)
        wd_scheduler.step(epoch+1)

        # record train loss
        history['epoch'].append(epoch)
        history['train_loss'].append(loss)
        if not args.use_ddp or local_rank == 0:
            writer.add_scalar('train_loss', loss, epoch)

        # **Validate** every eval_freq
        if epoch % b_setup['eval_freq'] == 0:
            avg_psnr, avg_ssim, val_loss = valid(val_loader, network, criterion)
            history['val_loss'].append(val_loss)
            history['psnr'].append(avg_psnr)
            history['ssim'].append(avg_ssim)

            if not args.use_ddp or local_rank == 0:
                # save best
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({
                        'cur_epoch': epoch+1,
                        'best_psnr': best_psnr,
                        'state_dict': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'wd_scheduler': wd_scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'history': history
                    }, os.path.join(save_dir, f"{args.model}.pth"))
                    
                    shutil.make_archive('/kaggle/working/saved_models', 'zip', '/kaggle/working/saved_models/')


                # periodic snapshot
                if epoch % 30 == 0:
                    torch.save({
                        'cur_epoch': epoch+1,
                        'best_psnr': best_psnr,
                        'state_dict': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'wd_scheduler': wd_scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'history': history
                    }, os.path.join(save_dir, f"{args.model}_epoch{epoch}.pth"))
                    
                    shutil.make_archive('/kaggle/working/saved_models', 'zip', '/kaggle/working/saved_models/')


                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('best_psnr',  best_psnr, epoch)
            if args.use_ddp:
                torch.distributed.barrier()
        else:
            # keep list alignment
            history['val_loss'].append(None)
            history['psnr'].append(None)
            history['ssim'].append(None)

        # **Dump CSV** every epoch
        if not args.use_ddp or local_rank == 0:
            df = pd.DataFrame(history)
            df.to_csv(os.path.join(save_dir, f"{args.model}_training_log.csv"), index=False)

    # 10) Done
    if not args.use_ddp or local_rank == 0:
        print("Done. Training log & checkpoints saved.")

if __name__ == '__main__':
    main()



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose

from utils.data_utils import get_loader
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse

from models.Trans import CONFIGS as CONFIGS_TM

import models.Trans as Trans

import time
from monai.metrics import hausdorff_distance, HausdorffDistanceMetric, SurfaceDiceMetric
from trainer import test_organ
from monai.transforms import KeepLargestConnectedComponent,RemoveSmallObjects

# Arguments
parser = argparse.ArgumentParser(description='SMIT+Swin-UNETR segmentation testing pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='unetr_test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/your data folder/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='./dataset/dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=10000, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--model_name', default='swin3D', type=str, help='model name')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=14, type=int, help='number of output channels')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--use_smart_cache',action='store_true',help='use SmartCacheDataset class')
parser.add_argument('--cache_num',default=10,type=int,help='number of items to cache in CacheDataset')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--surface_threshold',default=1.0,type=float,help='Surface Dice threshold')



def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = True
    args.get_encodings=False
    loader = get_loader(args)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if 1:
        
        
        config = CONFIGS_TM['Trans-Small_Unetr'] # this one add the patch size from 4 to 2
        model = Trans.Trans_Unetr(config,out_channels=args.out_channels)

        calc_param = lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)
        
        print (model)
        print(f"Model param: {calc_param(model) / 1e6 : .2f} M")

        if args.resume_ckpt:
            

            pretrained_path = os.path.normpath('/Morfeus/Nihil/SMIT/Pre_trained/pre_train_weight.pt')

            pretrained_dict = torch.load(pretrained_path)
            
            for key in list(pretrained_dict.keys()):
                pretrained_dict[key.replace('module.', '')] = pretrained_dict.pop(key)
            model.transformer.load_state_dict(pretrained_dict,strict=False)


            print('info: successfully Pretrained Weights Succesfully Loaded from !', pretrained_path)
        else:
            print ('info: No pretraining mode')
        if args.resume_jit:
            if not args.noamp:
                print('Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))

    post_pred = Compose([

                            AsDiscrete(argmax=True,
                           to_onehot=args.out_channels),
                        KeepLargestConnectedComponent(applied_labels=[1],is_onehot=True),
                            ])

    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=args.sw_batch_size,
                            predictor=model,
                            overlap=args.infer_overlap)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint '{}'".format(args.checkpoint))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.gpu],
                                                            output_device=args.gpu,
                                                            find_unused_parameters=False)
    test_organ(model,loader,args,model_inferer,post_pred)



     

if __name__ == '__main__':
    main()

# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import timm
from timm.models.layers import trunc_normal_
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torchvision.models as models
import util.misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn as nn
from datasets.image_datasets import build_image_dataset
from engine_finetune import train_one_epoch, evaluate
import models.vit_image1 as vit_image
import AveragePrecision
from torch.autograd import Variable
from timm.scheduler import CosineLRScheduler
import utils
from timm.utils import NativeScaler, get_state_dict, ModelEma
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from datasets.threeaugment import new_data_aug_generator

step_size = 10

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition for image classification', add_help=False)
    # batchsize
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # epochs
    parser.add_argument('--epochs', default=100, type=int)

    # 模型选择
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # EMA
    parser.add_argument('--model-ema', action='store_true', help='启用EMA模型更新')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema', help='禁用EMA模型更新')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='EMA衰减因子，默认为0.99996')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='强制将EMA模型存储和更新在CPU中')

    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    # 学习率调度
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='学习率调度器，默认使用"cosine"调度')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='起始学习率')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=[0.1, 0.2], metavar='pct, pct',
                        help='学习率噪声的启用/禁用周期')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='学习率噪声的百分比限制，默认值为0.67')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='学习率噪声的标准差，默认值为1.0')
    parser.add_argument('--warmup-lr', type=float, default=0.000001, metavar='LR',
                        help='热身学习率，默认值为1e-6')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='学习率热身的轮数，默认值为5')
    parser.add_argument('--min-lr', type=float, default=0.00001, metavar='LR',
                        help='学习率调度器的最低学习率，默认值为1e-5')
    parser.add_argument('--decay-epochs', type=float, default=50, metavar='N',
                        help='学习率衰减的周期，默认值为30')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='学习率在最低学习率后冷却的轮数，默认值为10')

    # 早停参数
    parser.add_argument('--early_stop_patience', default=20, type=int,
                        help='如果连续多少个epoch验证集mAP没有提升，则提前停止训练')

    # 数据增强相关
    parser.add_argument('--ThreeAugment', default=True, action='store_true', help='启用三种数据增强')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='颜色抖动因子，默认值为0.4')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='使用AutoAugment策略，默认值为"rand-m9-mstd0.5-inc1"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='标签平滑因子，默认值为0.1')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='训练图像插值方法，默认使用"bicubic"')

    # 微调参数
    parser.add_argument('--finetune', default='/home/zhanghongyu/hehaoran/xzh/plotadapter/pth/mae_pretrain_vit_b.pth', help='预训练路径')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='使用分类token代替全局池化进行分类')

    # 数据集相关参数
    parser.add_argument('--data_path', default='/home/zhanghongyu/hehaoran/xzh/Rdata', type=str,
                        help='数据集路径')
    parser.add_argument('--nb_classes', default=13, type=int,
                        help='分类类别数')

    # 输出目录
    parser.add_argument('--output_dir', default='./output_dir',
                        help='保存路径，若为空则不保存')
    parser.add_argument('--log_dir', default=None,
                        help='TensorBoard日志路径')
    parser.add_argument('--device', default='cuda',
                        help='用于训练和测试的设备')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='起始训练轮数')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='在DataLoader中固定CPU内存，以提高数据传输效率')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # 自定义配置
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path的比率（默认值：0.0）')

    # AdaptFormer相关参数
    parser.add_argument('--ffn_adapt', default=True, action='store_true', help='是否启用AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='瓶颈层的维度')
    parser.add_argument('--vpt', default=False, action='store_true', help='是否启用VPT')
    parser.add_argument('--vpt_num', default=16, type=int, help='VPT提示的数量')
    parser.add_argument('--fulltune', default=False, action='store_true', help='是否进行完整的微调')

    return parser


def main(args):

    def Load_Image_Information(path):
        image_Root_Dir = r'/home/zhanghongyu/hehaoran/xzh/olddata/python/multi_label'
        iamge_Dir = os.path.join(image_Root_Dir, path)
        return Image.open(iamge_Dir).convert('RGB')

    class my_Data_Set(nn.Module):
        def __init__(self, txt, transform=None, target_transform=None, loader=None):
            super(my_Data_Set, self).__init__()
            fp = open(txt, 'r')
            images = []
            labels = []
            for line in fp:
                line.strip('\n')
                line.rstrip()
                information = line.split()
                images.append(information[0])
                labels.append([float(l) for l in information[1:len(information)]])
            self.images = images
            self.labels = labels
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, item):
            imageName = self.images[item]
            label = self.labels[item]
            imageName = imageName[:-3] + 'png'
            image = self.loader(imageName)
            if self.transform is not None:
                image = self.transform(image)
            label = torch.FloatTensor(label)
            return image, label

        def __len__(self):
            return len(self.images)

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196]),
            transforms.RandomErasing()
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196])
        ])
    }

    dataset_train = my_Data_Set(
        r'/home/zhanghongyu/hehaoran/xzh/olddata/python/train_label.txt',
        transform=data_transforms["train"],
        loader=Load_Image_Information
    )
    dataset_val = my_Data_Set(
        r'/home/zhanghongyu/hehaoran/xzh/olddata/python/val_label.txt',
        transform=data_transforms["val"],
        loader=Load_Image_Information
    )

    label_counts = Counter()
    for label in dataset_train.labels:
        for i, val in enumerate(label):
            if val == 1:
                label_counts[i] += 1

    total_samples = len(dataset_train.labels)
    class_weights = {key: total_samples / (len(dataset_train.labels) * count) for key, count in label_counts.items()}

    sample_weights = []
    for label in dataset_train.labels:
        weight = 0
        for i, val in enumerate(label):
            if val == 1:
                weight += class_weights[i]
        sample_weights.append(weight)

    sample_weights = torch.tensor(sample_weights)
    sampler_train = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    tuning_config = EasyDict(
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )

    if args.model.startswith('vit'):
        model = vit_image.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
            drop_path_rate=args.drop_path,
            tuning_config=tuning_config,
        )
    else:
        raise NotImplementedError(args.model)

    if True:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=args.warmup_epochs,
        cycle_limit=1,
    )

    class MultiLabelFocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(MultiLabelFocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return F_loss.mean()

    criterion = MultiLabelFocalLoss(alpha=1, gamma=2)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_acc = 0.0
    best_tensor = None
    no_improve_epochs = 0
    patience = args.early_stop_patience

    for epoch in range(args.epochs):
        model.train()
        batch_size_start = time.time()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(data_loader_train):
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            assert torch.all(torch.isfinite(inputs)), "Inputs contain NaN or Inf"
            assert torch.all(torch.isfinite(labels)), "Labels contain NaN or Inf"

            optimizer.zero_grad()

            result = model(inputs)
            loss = criterion(result, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % step_size == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Iter [{i + 1}/{len(data_loader_train)}] Loss: {running_loss / step_size:.4f}")
                running_loss = 0.0

        lr_scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch + 1}/{args.epochs}] Training time: {time.time() - batch_size_start:.4f} seconds')
        print(f"Learning Rate: {current_lr:.6f}")

        if args.model_ema:
            model_ema.update(model)

        model.eval()
        ap_meter = AveragePrecision.AveragePrecisionMeter(difficult_examples=False)

        with torch.no_grad():
            for j, (inputs, labels) in enumerate(data_loader_val):
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)

                result = model(inputs)
                ap_meter.add(result.data, labels)

        ap_values = ap_meter.value()
        print("Average Precision per class: ", 100 * ap_values)
        map = 100 * ap_values.mean()
        print(f"Mean Average Precision (mAP): {map:.4f}")

        if map > best_acc:
            best_acc = map
            best_tensor = ap_meter.value()
            no_improve_epochs = 0

            if args.output_dir:
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path} at epoch {epoch + 1}")
        else:
            no_improve_epochs += 1
            print(f"No mAP improvement for {no_improve_epochs} epoch(s)")

        print(f"Validation mAP: {map:.4f}")
        print(f"Best mAP: {best_acc:.4f}")

        if map >= 99:
            print(f"mAP >= 99, stopping training at epoch {epoch + 1}")
            break

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered: no mAP improvement for {patience} consecutive epochs. Stop at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training finished, total time: {total_time_str}")
    print(f"Tensor corresponding to highest mAP: {best_tensor}")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
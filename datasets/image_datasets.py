import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset  # 正确导入 Dataset


# 自定义图像加载函数
def Load_Image_Information(path):
    image_Root_Dir = r'/home/zhanghongyu/hehaoran/projects/xzh/newdata/multi_label'
    image_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(image_Dir).convert('RGB')


# 自定义数据集类，继承自 Dataset 而非 nn.Module
class my_Data_Set(Dataset):  # 应该继承自 Dataset
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开文件并读取图像和标签信息
        with open(txt, 'r') as fp:  # 使用 with 语句自动关闭文件
            images = []
            labels = []
            for line in fp:
                line = line.strip()  # 去除换行符
                information = line.split()  # 按空格分割
                images.append(information[0])  # 图像路径
                labels.append([float(l) for l in information[1:]])  # 标签（转为浮点型列表）
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        image_name = self.images[item]
        label = self.labels[item]
        image_name = image_name[:-3] + 'png'  # 假设文件扩展名是 .jpg 改成 .png
        image = self.loader(image_name)  # 使用自定义的加载函数
        if self.transform is not None:
            image = self.transform(image)  # 应用数据预处理
        label = torch.FloatTensor(label)  # 将标签转为 FloatTensor
        return image, label

    def __len__(self):
        return len(self.images)


# 定义数据预处理方法
def build_image_dataset(args):
    # 自定义的数据集路径和预处理
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
    
    # 创建自定义数据集实例
    train_txt = os.path.join(args.data_path, 'train_label.txt')
    val_txt = os.path.join(args.data_path, 'val_label.txt')
    
    train_data = my_Data_Set(train_txt, transform=data_transforms["train"], loader=Load_Image_Information)
    val_data = my_Data_Set(val_txt, transform=data_transforms["val"], loader=Load_Image_Information)
    
    # 创建 DataLoader
    batch_size = args.batch_size  # 从参数获取批量大小
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16)
    
    nb_classes = 13  # 假设你的数据集有13个类别，根据实际情况调整
    
    return train_loader, val_loader, nb_classes

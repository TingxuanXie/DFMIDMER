#导入视频数据：给定‘class file_name label�?-->'img_path label'
#write by syx in 2020/12/11
import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from natsort import natsorted
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import transforms.temporal_transforms as TT


class load_data(Dataset):
    def __init__(self, file, root, transform=None, temporal_transform=None):
        self.file = file
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.root = root
        self.prepare_file()

    def prepare_file(self):
        lines = list(open(self.file))  # len(lines) = 训练样本数；列表中的元素是txt中每一
        video_path, label = self.load_p_data(lines)

        self.video = video_path
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return len(self.video)

    # 输入：index
    # 输出：（1）列表：�?个子列表,分别是rgb和flow�? --> 子列表：各含�?6�?3,128,128)的tensor
    # 输出：（2）一个tensor标量
    # 输出：（3）index
    def get_data(self, index):
        image_path = self.video[index]  # 两个列表 --> 每个列表�?6个图片路�?
        image_path = self.temporal_transform(image_path)
        length = len(image_path)
        video = []
        for i in range(length):
            # 以RGB形式打开
            img = Image.open(image_path[i]).convert('RGB')
            # 图片transform
            img = self.transform(img)
            video.append(img)
        video = torch.stack(video, 0)  # �?6,3,256,256�?
        # megre_video.append(video)
        label = self.label[index]
        # return megre_video, label, index
        video = video.permute(1, 0, 2, 3)
        return video, label, index

    # 输入：列表，存放相对路径
    # 输出:�?�?若干列表 --> 每个子列表有16个图片的路径
    # 输出:�?）列表，存放184个数字，都是标签
    def load_p_data(self, video):
        video_path = []
        label_path = []
        for video_id in video:  # 遍历列表
            path = video_id.strip().split(' ')
            v_flow_path = self.root + '/' + path[0] + '/'
            flow_img_path = glob(os.path.join(v_flow_path, '*'))
            v_label = int(path[1].strip())

            # 变成自然排序
            flow_img_path = natsorted(flow_img_path)
            video_path.append(flow_img_path)
            label_path.append(v_label)
        label_path = np.array(label_path)
        return video_path, label_path



if __name__=='__main__':

    # 孙国荃实�?
    flow_root = r'F:\sunGuoQuan\experiment\syx\16_sdu_flow'
    file = r'F:\sunGuoQuan\experiment\syx\train_list_SDUwjc_sortbysample.txt'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    temporal_transform = TT.TemporalRandomCrop(size=8, stride=4)
    video_data = load_data(file=file, root=flow_root, transform=transform, temporal_transform=temporal_transform)

    train_loader=DataLoader(
        dataset=video_data ,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    for idx, (inputs, label, index) in enumerate(train_loader):
        print('idx', idx, 'inputs', inputs.shape, 'label', label.shape)


    # print(np.array(video_data[0]).shape)






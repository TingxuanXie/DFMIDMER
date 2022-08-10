from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from glob import glob
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import transforms.temporal_transforms as TT


class load_data_id(Dataset):
    def __init__(self, file, root, spatial_transform=None, temporal_transform=None):
        self.file = file
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.root = root

        txt_lines_list = list(open(self.file))
        video_path_list = []
        label_list = []
        for path in txt_lines_list:
            path_list = path.strip().split(' ')
            video_absolute_path = self.root + '/' + path_list[0] + '/'
            image_absolute_path_list = glob(os.path.join(video_absolute_path, '*'))
            video_label = int(path_list[1].strip())

            # natural sort 自然排序=zi_ran_pai_xu
            image_absolute_path_list = natsorted(image_absolute_path_list)
            video_path_list.append(image_absolute_path_list)
            label_list.append(video_label)
        label_list = np.array(label_list)
        self.video_path_list = video_path_list
        self.label_list = torch.from_numpy(label_list)


    def __getitem__(self, index):
        image_absolute_path_list = self.video_path_list[index]
        # print(image_absolute_path_list)
        image_absolute_path_list = self.temporal_transform(image_absolute_path_list)
        images_num = len(image_absolute_path_list)
        images_list = []
        for i in range(images_num):
            image_RGB = Image.open(image_absolute_path_list[i]).convert('RGB')
            image_RGB = self.spatial_transform(image_RGB)
            images_list.append(image_RGB)
        # list2tensor
        images = torch.stack(images_list, 0)
        label = self.label_list[index]
        images = images.permute(1, 0, 2, 3)

        return images, label


    def __len__(self):
        return len(self.video_path_list)


if __name__=='__main__':
    file = r'F:\sgqbishe2\data/SDU_train_label_0331_id.txt'
    root = r'F:\sunGuoQuan\SDU/SDU'
    spatial_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    temporal_transform = TT.TemporalRandomCrop(size=8, stride=4)
    video_data = load_data_id(file=file, root=root,
                           spatial_transform=spatial_transform,
                           temporal_transform=temporal_transform)

    train_loader = DataLoader(
        dataset=video_data,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        drop_last=True)

    for idx, (inputs, label) in enumerate(train_loader):
        print('idx', idx, 'inputs', inputs.shape, 'label', label.shape)

    # for inputs, label in video_data:
    #     print('inputs', inputs.shape, 'label', label)
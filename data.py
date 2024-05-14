from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np

class CelebA(data.Dataset):
    """
    A custom dataset class for the CelebA dataset.

    Args: 
        data_path (str): Path to the CelebA dataset.
        attr_path (str): Path to the attribute file.
        image_size (int): Image size.
        mode (str): Mode of the dataset (train, valid, or test).
        selected_attrs (list): Selected attributes for the CelebA dataset.
        stargan_selected_attrs (list): Selected attributes for StarGAN.

    Returns:
        img (Tensor): Images.
        att (Tensor): Attributes.
        label (Tensor): Labels.
    """
    def __init__(self, args, data_path, attr_path, image_size, mode, selected_attrs, stargan_selected_attrs):
        super(CelebA, self).__init__()
        self.args = args
        self.data_path = data_path
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.stargan_selected_attrs = stargan_selected_attrs
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str_)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int64)
        
        if mode == 'test':
            if self.args.global_settings.num_test is not None:
                self.images = images[-self.args.global_settings.num_test:]
                self.labels = labels[-self.args.global_settings.num_test:]
            else:
                self.images = images[-64:]
                self.labels = labels[-64:]
        elif mode == 'default-test':
            self.images = images[182637:]
            self.labels = labels[182637:]
        elif mode == 'train':
            self.images = images[:self.args.global_settings.num_test]
            self.labels = labels[:self.args.global_settings.num_test]
        elif mode == 'default-train':
            self.images = images[:182000]
            self.labels = labels[:182000]
        elif mode == 'default-valid':
            self.images = images[182000:182637]
            self.labels = labels[182000:182637]
        else:
            raise Exception('Invalid mode')
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                                       
        self.length = len(self.images)

        # stargan
        self.attr2idx = {}
        self.idx2attr = {}
        self.test_dataset = []
        self.preprocess()

    def preprocess(self):
        """
        Preprocesses the CelebA dataset by reading attribute values from a file,
            creating mappings between attribute names and indices, and constructing
            the test dataset with filenames and corresponding labels.

            Returns:
                None        
        """
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[182637:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.stargan_selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            self.test_dataset.append([filename, label])
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """
        Returns the image, attribute, and label.
        """
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        filename, label = self.test_dataset[index]

        return img, att, torch.FloatTensor(label)
        
    def __len__(self):
        """
        Returns the length of the dataset. Namely, the number of images.
        """
        return self.length




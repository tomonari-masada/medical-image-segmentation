import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image


class DatasetFromFolder(data.Dataset):
    def __init__(self, tag='train', input_transform=None, label_transform=None):
        super(DatasetFromFolder, self).__init__()

        image_dir = tag + '/image/'
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)]

        label_dir = tag + '/label/'
        self.label_filenames = [join(label_dir, x) for x in listdir(label_dir)]

        self.input_transform = input_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        input = Image.open(self.image_filenames[index])
        label = Image.open(self.label_filenames[index])

        if self.input_transform:
            input = self.input_transform(input)
        if self.label_transform:
            label = self.label_transform(label)

        return input, label

    def __len__(self):
        return len(self.image_filenames)

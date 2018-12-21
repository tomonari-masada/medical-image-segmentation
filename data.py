from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import zipfile
from torchvision.transforms import Compose, Lambda, ToTensor

from dataset import DatasetFromFolder


def download_data(tag='train'):

    if not exists(tag):
        makedirs(tag)
        url = 'https://github.com/mitmul/chainer-handson/releases/download/SegmentationDataset/' + tag + '.zip'
        print('# downloading url ', url)

        data = urllib.request.urlopen(url)

        file_path = url.split('/')[-1]
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print('# Extracting data')
        with zipfile.ZipFile(file_path) as zip_ref:
            zip_ref.extractall()

        remove(file_path)
    return tag


def input_transform():
    return Compose([
        ToTensor(),
    ])


def label_transform():
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.mul_(255.)),
    ])


def get_data(tag='train'):
    download_data(tag)
    data_dir = tag
    return DatasetFromFolder(data_dir,
                             input_transform=input_transform(),
                             label_transform=label_transform())

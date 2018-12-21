import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def show_image(index, tag='train'):
    imgid = '{:03d}'.format(index) 
    img = np.asarray(Image.open(tag + '/image/'+ imgid + '.png'))
    label = np.asarray(Image.open(tag + '/label/' + imgid + '.png'))

    fig, axes = plt.subplots(1, 2)
    axes[0].set_axis_off()
    axes[0].imshow(img, cmap='gray')
    axes[1].set_axis_off()
    axes[1].imshow(label, cmap='gray')
    plt.show()
    return

if __name__ == '__main__':
    show_image(0, tag='train')

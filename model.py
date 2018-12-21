import torch
import torch.nn as nn
import torch.nn.init as init


class FullyConvolutionalNetwork(nn.Module):
    def __init__(self, out_h, out_w, n_class=1):
        super(FullyConvolutionalNetwork, self).__init__()

        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)
        
        self.conv1 = nn.Conv2d(1, 64, 5, 2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 128, 1, 1, 0)
        self.deconv6 = nn.ConvTranspose2d(128, n_class, 32, 16, 8)
        self.out_h = out_h
        self.out_w = out_w

        self._initialize_weights()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pooling(h)

        h = self.relu(self.conv2(h))
        h = self.pooling(h)

        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.conv5(h)
        h = self.deconv6(h)
        
        return h.reshape(x.shape[0], 1, h.shape[2], h.shape[3])

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight, init.calculate_gain('relu'))

if __name__ == '__main__':
    print(FullyConvolutionalNetwork(256, 256)(torch.zeros(1, 1, 256, 256)).shape[2:])

import torch.nn as nn


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 3 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.pixel_shuffle(self.conv4(x)))
        return x

    def _initialize_weights(self):
        nn.init.orthogonal(self.conv1.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv2.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv3.weight, nn.init.calculate_gain('relu'))
        nn.init.orthogonal(self.conv4.weight)


if __name__ == "__main__":
    model = Net(upscale_factor=3)
    print(model)

"""""""""
Convolutional auto-encoder
"""""""""
import torch
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, in_channel=3):
        super(CAE, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.down(x)
    
    def decode(self, z):
        return self.up(z)

    def forward(self, x):
        z = self.down(x)
        return self.up(z)


if __name__ == '__main__':
    x = torch.ones(16,1,80,80)
    model = CAE(in_channel=1)
    print(x.shape)
    print(model(x).shape)

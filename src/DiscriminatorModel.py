import torch.nn as nn

class DiscriminatorModel(nn.Module):
    def __init__(
        self,
        channels=1,
        image_size=16
    ):
        super(DiscriminatorModel, self).__init__()

        scaled_image_size = image_size//4

        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(scaled_image_size * scaled_image_size * 128, 1),

        )

    def forward(self, x):
        return self.main(x)
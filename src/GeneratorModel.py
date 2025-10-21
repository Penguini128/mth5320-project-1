import torch
import torch.nn as nn

class GeneratorModel(nn.Module):
    def __init__(
            self,
            noise_dims=64,
            generated_image_size=16,
            channels=1,
            sharpen_output=False,
            sharpen_factor=8,
        ):
        super(GeneratorModel, self).__init__()
        self.sharpen_output = sharpen_output
        self.sharpen_factor = sharpen_factor

        starting_dims = generated_image_size // 4

        self.main = nn.Sequential(
            nn.Linear(noise_dims, starting_dims**2 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, starting_dims, starting_dims)),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 5, stride=2,
                               padding=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        result = self.main(x)
        if not self.training and  self.sharpen_output:
            result = torch.round(result * self.sharpen_factor) / self.sharpen_factor
        return result

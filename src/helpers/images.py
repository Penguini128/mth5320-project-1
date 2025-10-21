import torch
from matplotlib import pyplot as plt
import os
import numpy as np

"""
Helper Function for Plotting Generated Images
"""
def generate_and_save_images(model, epoch, noise, params, output_folder):
    model.eval()
    num_channels = 1 if params['TRAINING_SPRITE_TYPE'] == 'grayscale' else 3
    with torch.no_grad():
        image_size = 16 if params['TRAINING_DATASET'] == 'sprite' else 28
        fake_images = model(noise).cpu()
        fake_images = fake_images.view(fake_images.size(0), num_channels, image_size, image_size)

        fig = plt.figure(figsize=params['GENERATE_EXAMPLES_DIMS'])
        for i in range(fake_images.size(0)):
            plt.subplot(params['GENERATE_EXAMPLES_DIMS'][0], params['GENERATE_EXAMPLES_DIMS'][1], i + 1)
            plt.imshow(np.clip(fake_images[i].permute(1, 2, 0), 0, 1), cmap='gray')
            plt.axis('off')

        plt.savefig(os.path.join(output_folder, f'images_at_epoch_{epoch+1}.png'))
        plt.show()
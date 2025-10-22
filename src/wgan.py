"""
wgan.py
"""

"""
Import Libraries
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from GeneratorModel import GeneratorModel
from DiscriminatorModel import DiscriminatorModel
import os
from helpers import *

"""
Define constants
"""
params = {
    'NOISE_DIMENSIONS' : 64,

    'TRAINING_DATASET' : 'sprite',
    'TRAINING_SPRITE_TYPE' : 'grayscale',
    'TRAINING_SPRITE_CATEGORY' : ['food', 'people', 'side_profiles', 'items'],
    'TRAINING_UNIQUES_ONLY' : True,

    'GENERATOR_LEARNING_RATE' : 0.0005,
    'GENERATOR_ADAM_BETAS' : (0.0, 0.98),
    'SHARPEN_GENERATOR_OUTPUT' : False,
    'SHARPEN_FACTOR' : 8,

    'DISCRIMINATOR_LEARNING_RATE': 0.0001,
    'DISCRIMINATOR_ADAM_BETAS': (0.0, 0.98),

    'NUM_EPOCHS' : 2048*10,
    'BATCH_SIZE' : 2048,
    'CRITIC_CYCLES' : 5,

    'GENERATE_EXAMPLES_DIMS' : (8, 8),
    'SAVE_EXAMPLES_PER_EPOCHS' : 128,
    'SAMPLE_CURVE_PER_EPOCHS' : 64
}

"""
Initialization
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Found device: {device}')

output_folder = generate_output_path()

"""
Load training data
"""
training_data = None
if params['TRAINING_DATASET'] == 'mnist':
    training_data = load_mnist_training_data()
elif params['TRAINING_DATASET'] == 'sprite':
    training_data = load_pixel_art_training_data(params['TRAINING_SPRITE_TYPE'], params['TRAINING_SPRITE_CATEGORY'])
else:
    raise ValueError(f'Invalid dataset "{params["TRAINING_DATASET"]}"')
assert training_data is not None

# Additionally make sure data is in the correct format to be passed to the model
if params['TRAINING_DATASET'] == 'sprite':
    if params['TRAINING_UNIQUES_ONLY']:
        training_data = np.unique(training_data, axis=0)
    if params['TRAINING_SPRITE_TYPE'] == 'rgb':
        training_data = torch.tensor(training_data).to(device).permute(0, 3, 1, 2)
    if params['TRAINING_SPRITE_TYPE'] == 'grayscale':
        training_data = torch.tensor(training_data).to(device).unsqueeze(1)
    print(training_data.shape)

"""
Instantiate generator and discriminator
"""
NUM_CHANNELS = 1 if params['TRAINING_SPRITE_TYPE'] == 'grayscale' else 3
generator_model = GeneratorModel(
    noise_dims=params['NOISE_DIMENSIONS'],
    channels=NUM_CHANNELS,
    generated_image_size=16 if params['TRAINING_DATASET'] == 'sprite' else 28,
    sharpen_output=params['SHARPEN_GENERATOR_OUTPUT'],
    sharpen_factor=params['SHARPEN_FACTOR'],
).to(device)
discriminator_model = DiscriminatorModel(
    channels=NUM_CHANNELS,
    image_size=16 if params['TRAINING_DATASET'] == 'sprite' else 28
).to(device)

generator_optimizer = optim.Adam(
    generator_model.parameters(),
    lr=params['GENERATOR_LEARNING_RATE'],
    betas=params['GENERATOR_ADAM_BETAS']
)
discriminator_optimizer = optim.Adam(
    discriminator_model.parameters(),
    lr=params['DISCRIMINATOR_LEARNING_RATE'],
    betas=params['DISCRIMINATOR_ADAM_BETAS']
)

"""
Load training data into DataLoader
"""

train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=params['BATCH_SIZE'],
    shuffle=True
)

"""
Execute Training Loop
"""
discriminator_losses = []
generator_losses = []

test_noise = torch.randn(int(np.prod(params['GENERATE_EXAMPLES_DIMS'])), params["NOISE_DIMENSIONS"],
                                 device=device)
# Loop for specified number of epochs
for epoch in range(params['NUM_EPOCHS']):
    cumulative_gen_loss = 0
    cumulative_dis_loss = 0
    for batch_index, batch in enumerate(train_loader):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        # Get and normalized batch data
        real_images = batch
        real_images = real_images.to(device).float()

        fake_images=None

        for i in range(params['CRITIC_CYCLES']):
            # Pass real data through discriminator
            discriminator_optimizer.zero_grad()
            real_outputs = discriminator_model(real_images)

            # Use noise to generate fake data, then pass fake data through discriminator
            noise = torch.randn(real_images.size(0), params['NOISE_DIMENSIONS'], device=device)
            fake_images = generator_model(noise).detach()
            fake_outputs = discriminator_model(fake_images)
            critic_loss = torch.mean(fake_outputs) - torch.mean(real_outputs)
            critic_loss.backward()
            # Update weights in discriminator
            discriminator_optimizer.step()

            for p in discriminator_model.parameters():
                p.data.clamp_(-0.01, 0.01)

        assert fake_images is not None

        # Use the same noise to generate same fake data, then update generator
        generator_optimizer.zero_grad()
        noise = torch.randn(real_images.size(0), params['NOISE_DIMENSIONS'], device=device)
        fake_images = generator_model(noise)
        fake_outputs = discriminator_model(fake_images)
        gen_loss = -torch.mean(fake_outputs)
        gen_loss.backward()
        # Update weights in generator
        generator_optimizer.step()

        # Cumulate loss
        cumulative_gen_loss += gen_loss.item()
        cumulative_dis_loss += critic_loss.item()

        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(train_loader):
            print(f'Epoch [{epoch+1}/{params["NUM_EPOCHS"]}], Step [{batch_index+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {critic_loss.item():.4f}, '
                  f'Generator Loss: {gen_loss.item():.4f}')

    if (epoch + 1) % params['SAVE_EXAMPLES_PER_EPOCHS'] == 0 or epoch + 1 == params['NUM_EPOCHS']:
        generate_and_save_images(generator_model, epoch, test_noise, params, output_folder)
        generator_model.train()

    if (epoch + 1) % params['SAMPLE_CURVE_PER_EPOCHS'] == 0:
        discriminator_losses.append([epoch, cumulative_dis_loss])
        generator_losses.append([epoch, cumulative_gen_loss])

discriminator_losses = np.array(discriminator_losses)
generator_losses = np.array(generator_losses)

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(discriminator_losses[:, 0].reshape(-1), discriminator_losses[:, 1].reshape(-1), label='Discriminator')
ax.plot(generator_losses[:, 0].reshape(-1), generator_losses[:, 1].reshape(-1), label='Generator')
plt.legend()
plt.savefig(os.path.join(output_folder, 'loss_plot.png'))
plt.show()
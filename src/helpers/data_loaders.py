import numpy as np
import torchvision
from torchvision import transforms

"""
Load Training Data
"""
def load_pixel_art_training_data(sprite_type, sprite_category):
    sprites = np.load('../data/sprites.npy')
    grayscale_sprites =  np.dot(sprites[..., :], [0.2989, 0.5870, 0.1140])

    sprites = (sprites - np.min(sprites)) / (np.max(sprites) - np.min(sprites))
    grayscale_sprites = (grayscale_sprites - np.min(grayscale_sprites)) / (np.max(grayscale_sprites) - np.min(grayscale_sprites))

    labels = np.load('../data/sprites_labels.npy')
    int_labels = np.argmax(labels, axis=1)

    data = {
        'rgb' : {
            'people' : sprites[int_labels == 0],
            'creatures' : sprites[int_labels == 1],
            'food' : sprites[int_labels == 2],
            'items' : sprites[int_labels == 3],
            'side_profiles' : sprites[int_labels == 4],
            'all' : sprites
        },
        'grayscale' : {
            'people': grayscale_sprites[int_labels == 0],
            'creatures': grayscale_sprites[int_labels == 1],
            'food': grayscale_sprites[int_labels == 2],
            'items': grayscale_sprites[int_labels == 3],
            'side_profiles': grayscale_sprites[int_labels == 4],
            'all': grayscale_sprites
        }
    }

    if isinstance(sprite_category, str):
        return data[sprite_type][sprite_category]

    gathered_data = []
    for category in sprite_category:
        found = data[sprite_type][category][:]
        np.random.shuffle(found)
        gathered_data.extend(found[:128])
    return gathered_data

def load_mnist_training_data():
    mnist_transform = transforms.Compose([
        transforms.ToTensor()  # Converts HxW [0,255] -> CxHxW [0,1]
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=mnist_transform)
    return train_dataset
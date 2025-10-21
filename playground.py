import numpy as np
import matplotlib.pyplot as plt

"""
0 - People
1 - Creatures
2 - Food
3 - Items
4 - Side Profiles
"""

sprites = np.load('data/sprites.npy')
grayscale_sprites =  np.dot(sprites[..., :], [0.2989, 0.5870, 0.1140])
labels = np.load('data/sprites_labels.npy')
int_labels = np.argmax(labels, axis=1)

people = sprites[int_labels == 0]
creatures = sprites[int_labels == 1]
food = sprites[int_labels == 2]
items = sprites[int_labels == 3]
side_profiles = sprites[int_labels == 4]

temp_sprites = sprites.reshape(sprites.shape[0], -1)
print(temp_sprites.shape)
unique = np.unique(temp_sprites, axis=0)
unique = unique.reshape(unique.shape[0], 16, 16, 3)
print(unique.shape)


def simple_plot(data, rows, columns, offset=0) -> None:
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows * 1.5))
    for i in range(rows):
        for j in range(columns):
            ax = axes[i, j]
            ax.axis('off')
            index = i * columns + j + offset*rows*columns
            if index >= data.shape[0]:
                continue
            ax.imshow(data[i * columns + j + offset*rows*columns], cmap='gray')


    plt.show()

simple_plot(unique, 40, 45, offset=0)

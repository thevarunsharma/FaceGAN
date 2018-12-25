import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

gen = load_model('generator.hdf5')

def augment_images(images):
    images = (images - images.min())/(images.max() - images.min())
    images = np.array(images*255, dtype='uint8')
    return images

def display_images(images):
    columns = 4
    rows = images.shape[0]//4+1
    fig = plt.figure(figsize=(9, 3*rows))
    # ax enables access to manipulate each of subplots
    ax = []
    for i in range(images.shape[0]):
        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_yticklabels([])
        ax[-1].set_xticklabels([])
        plt.imshow(images[i], cmap='gray')

    plt.show()  # finally, render the plot

while True:
    print("\n"+"="*80)
    batch_size = int(input("How many faces do you want me to generate?: "))
    code = np.random.randn(batch_size, 256)
    images = gen.predict(code)
    images = augment_images(images)
    display_images(images)
    ch = input('Do you want to generate more?([y]/n) ')
    if ch.strip().lower()!='y':
        break
print("Bye!")

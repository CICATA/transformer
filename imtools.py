import os
from PIL import Image  # or you can use the keras one to load images
import numpy as np

def load_dataset(top_dir, width, height):
    images_dataset = []
    for root, dirs, files in os.walk(top_dir):
        for name in files:
            # print(os.path.join(root, name))
            # img = np.array(Image.open(os.path.join(root, name)).resize((width, height), Image.ANTIALIAS))[:, :, 0]
            img = np.array(Image.open(os.path.join(root, name)).resize((width, height), Image.ANTIALIAS))
            images_dataset.append(img)
    return np.stack(images_dataset, axis=0)


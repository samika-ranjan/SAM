import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def readAndStoreImages(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images.append(img)
    print(f"Total images loaded: {len(images)}")
    return np.array(images)

# Example usage
image = readAndStoreImages('/content/SAM/images/fiveimagepngs/')
segmask = readAndStoreImages('/content/SAM/images/fiveimagemasks/')
segmask = segmask[:, :, :, 0]

isomsk = np.concatenate((
    np.where(segmask[0:5, :, :] == 1, 1, 0),
    np.where(segmask[5:20, :, :] == 4, 1, 0)
), axis=0)

i = 12
plt.imshow(image[i], cmap='gray')
plt.imshow(isomsk[i], alpha=0.5)
plt.show()

# Sanity checks
print(isomsk.shape)
print(type(image))
print(np.shape(image))

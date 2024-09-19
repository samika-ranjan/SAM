import numpy as np
import matplotlib.pyplot as plt

def generate_sam_masks(image, isomsk, predictor):
    """
    Function to generate SAM masks for all images using centroids.
    """
    loopover = np.arange(len(image))
    sam_generated_mask = np.array([process_sam_mask(image[id], getCentroids(isomsk[id]), predictor) for id in loopover])
    return sam_generated_mask

def visualize_sam_masks(image, sam_generated_mask, centroid_coords=None):
    """
    Function to visualize each image with its SAM-generated mask.
    Optionally plots centroids on the images.
    """
    loopover = np.arange(len(image))
    for i in range(len(loopover)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image[loopover[i]])
        plt.imshow(sam_generated_mask[i], alpha=0.5)
        if centroid_coords is not None:
            plt.plot(centroid_coords[1], centroid_coords[0], 'ro')  # Plot centroid
        plt.show()

def convert_to_binary_masks(sam_generated_mask):
    """
    Converts SAM-generated masks to binary masks.
    """
    binary_sam_masks = np.where(sam_generated_mask[:, :, :] == 0, 0, 1)
    return binary_sam_masks

def sanity_check_binary_masks(image, binary_sam_masks):
    """
    Performs a sanity check by printing unique values of the binary masks and displaying them.
    """
    loopover = [2, 3, 4, 5, 18]
    for i in range(len(loopover)):
        print(np.unique(binary_sam_masks[i]))

        plt.figure(figsize=(10, 10))
        plt.imshow(image[loopover[i]])
        plt.imshow(binary_sam_masks[i], alpha=0.5)
        plt.show()

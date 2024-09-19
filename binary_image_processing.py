import scipy.ndimage
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Function to plot the matrix
def plotttt(matrix, title="Original Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to create a binary image for a specific class of objects
def create_binary_image(matrix, target):
    binary_image = np.where(matrix == target, 1, 0)
    return binary_image

# Function to find connected components in a binary matrix
def find_connected_components(matrix):
    labeled_matrix, num_features = scipy.ndimage.label(matrix)
    labeled_matrix = np.asarray(labeled_matrix, dtype=int)
    return labeled_matrix, num_features

# Function to read a matrix from a file
def read_matrix_from_file(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            matrix.append(list(map(float, line.split())))
        return np.array(matrix)

# Function to generate individual images of each connected component
def generate_single_images(binary_image):
    labeled_matrix, num_features = scipy.ndimage.label(binary_image)
    component_images = []
    for i in range(1, num_features + 1):
        component_image = np.where(labeled_matrix == i, 1, 0)
        component_images.append(component_image)
    return component_images

# Function to crop a binary image around the region of interest
def crop_image(binary_image, margin=5):
    non_zero_coords = np.argwhere(binary_image)
    top_left = non_zero_coords.min(axis=0)
    bottom_right = non_zero_coords.max(axis=0)

    start_x = max(top_left[0] - margin, 0)
    start_y = max(top_left[1] - margin, 0)
    end_x = min(bottom_right[0] + margin + 1, binary_image.shape[0])
    end_y = min(bottom_right[1] + margin + 1, binary_image.shape[1])

    return binary_image[start_x:end_x, start_y:end_y]

# Function to process all files in a directory for a given class of objects
def process_all_files(directory_path, objcls, singleorno=False):
    results = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            matrix = read_matrix_from_file(file_path)
            for particleclass in objcls:
                bin = create_binary_image(matrix, particleclass)
                if singleorno:
                    singleimages = generate_single_images(bin)
                    for sing in singleimages:
                        labeled_matrix, num_features = find_connected_components(sing)
                        pnt = 'img' + str(len(results) + 1) + '.txt'
                        results[pnt] = {
                            'file_name': file_name,
                            'labeled_matrix': labeled_matrix,
                            'num_features': num_features,
                            'particle_class': particleclass
                        }
                else:
                    print(f"Processing {file_name}: {particleclass}")
                    labeled_matrix, num_features = find_connected_components(bin)
                    pnt = 'img' + str(len(results) + 1) + '.txt'
                    results[pnt] = {
                        'file_name': file_name,
                        'labeled_matrix': labeled_matrix,
                        'num_features': num_features,
                        'particle_class': particleclass
                    }
                    print(results)
            print(f"Processed {file_name}.")
    return results

# Example usage
directory_path = '/content/images/bws'
results = {}
results['iso'] = process_all_files(directory_path, [1], False)

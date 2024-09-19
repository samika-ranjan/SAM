from skimage.measure import regionprops, label
import numpy as np
import matplotlib.pyplot as plt

def getCentroids(rawimg):
    centroid = [prop.centroid for prop in regionprops(label(rawimg))]
    return [(round(x), round(y)) for x, y in np.array(centroid)]

def process_sam_mask(rawimg, centroid, predictor):
    predictor.set_image(rawimg)
    print('processing')
    iso = rawimg[:, :, 0] * 0
    for item in centroid:
        input_point = np.array([[item[1], item[0]]])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        iso = iso + masks[0] 
    return iso

# Example usage for a specific image
imgid = 0
predictor.set_image(image[imgid])

# Get centroids and process SAM mask
centroid = getCentroids(isomsk[imgid])
print(centroid)

# Display image with centroids
plt.figure(figsize=(10, 10))
plt.imshow(image[imgid])

iso = isomsk[imgid] * 0
for item in centroid:
    input_point = np.array([[item[1], item[0]]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    plt.plot(item[1], item[0], 'ro')  # Plot centroid
    iso = iso + masks[0]

# Additional input points for the mask
input_point = np.array([[1358, 983], [1438, 983]])
input_label = np.array([0, 1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)
plt.plot(1358, 983, 'ro')  # Plot additional centroid
iso = iso + masks[0]

# Display the final mask
plt.imshow(iso, alpha=0.5)
plt.show()

# Compute logical operations on masks
samxorsmart = np.logical_xor(isomsk[imgid], iso)
andand = np.logical_and(iso, iso)

# Print unique values and save mask to a file
print(np.unique(iso))
print(np.unique(andand))
plt.imshow(andand)
np.savetxt('1005smartiso.txt', andand, fmt='%d')

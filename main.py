import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import hdbscan
import argparse



def cluster_bounding_boxes_hdbscan(bboxes, 
                                   min_cluster_size=1, 
                                   min_samples=1):
    """
    Cluster bounding boxes using HDBSCAN.
    Returns an array of labels (same length as bboxes).
    """
    # Convert (x, y, w, h) to feature vectors
    data = []
    for (x, y, w, h) in bboxes:
        centerX = x + w / 2
        centerY = y + h / 2
        data.append([centerX, centerY, w, h])
    
    data = np.array(data)
    
    # Fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples)
    labels = clusterer.fit_predict(data)
    
    return labels


def limit_clusters(labels, max_clusters=4, bboxes=None):
    """
    Merge smaller clusters until we have at most `max_clusters`.
    `bboxes` can be used if you want to merge by spatial distance.
    """
    labels = np.array(labels)
    # Identify unique clusters (ignore noise label = -1 for merging)
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    n_clusters = len(unique_clusters)
    
    if n_clusters <= max_clusters:
        return labels  # No need to merge
    
    # Optional: Use bounding box centers for cluster centroid calculations
    if bboxes is not None:
        centers = np.array([
            (x + w/2, y + h/2) for (x, y, w, h) in bboxes
        ])
    else:
        # If no bboxes are provided, we can't do spatial merging easily
        # We'll just merge based on cluster size alone.
        centers = None
    
    # Count how many points in each cluster
    cluster_sizes = {}
    for c in unique_clusters:
        cluster_sizes[c] = np.sum(labels == c)
    
    # Sort clusters by size (ascending)
    clusters_sorted_by_size = sorted(unique_clusters, key=lambda c: cluster_sizes[c])
    
    # Merge the smallest clusters first
    while len(clusters_sorted_by_size) > max_clusters:
        smallest_cluster = clusters_sorted_by_size[0]
        
        # Find a cluster to merge into. Let's pick the *closest* bigger cluster by centroid distance.
        # 1. Compute centroid of the smallest cluster
        if centers is not None:
            idx_small = np.where(labels == smallest_cluster)[0]
            centroid_small = centers[idx_small].mean(axis=0)
            
            # 2. Find the nearest centroid among bigger clusters
            min_dist = float('inf')
            merge_target = None
            
            for c in clusters_sorted_by_size[1:]:  # skip the smallest cluster itself
                idx_big = np.where(labels == c)[0]
                centroid_big = centers[idx_big].mean(axis=0)
                dist = np.linalg.norm(centroid_small - centroid_big)
                if dist < min_dist:
                    min_dist = dist
                    merge_target = c
            
            # Merge smallest cluster into merge_target
            labels[idx_small] = merge_target
        else:
            # If we don't have centers, just pick the largest cluster
            largest_cluster = clusters_sorted_by_size[-1]
            labels[labels == smallest_cluster] = largest_cluster
        
        # Recompute cluster list
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        cluster_sizes = {c: np.sum(labels == c) for c in unique_clusters}
        clusters_sorted_by_size = sorted(unique_clusters, key=lambda c: cluster_sizes[c])
    
    return labels


def visualize_clusters_translucent(image, bboxes, labels, alpha=0.3):
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    # Generate a color for each cluster
    unique_labels = np.unique(labels)
    color_map = {}
    for c in unique_labels:
        if c == -1:
            # Noise (draw in gray)
            color_map[c] = (128, 128, 128)
        else:
            # Random color
            color_map[c] = tuple([random.randint(0, 255) for _ in range(3)])
    
    # Copy the original image for final visualization
    viz = image.copy()
    # Create an overlay for the filled rectangles
    overlay = viz.copy()
    
    # 1) Draw filled rectangles on the overlay
    for (bbox, label) in zip(bboxes, labels):
        (x, y, w, h) = bbox
        color = color_map[label]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # -1 fills the rectangle
    
    # 2) Blend the overlay with the base image to achieve translucency
    cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
    
    # 3) (Optional) Draw an outline on top if you still want to see the bounding box edges
    for (bbox, label) in zip(bboxes, labels):
        (x, y, w, h) = bbox
        color = color_map[label]
        cv2.rectangle(viz, (x, y), (x + w, y + h), color, 2)
    
    # Plot the results
    plt.figure(figsize=(15, 7))
    
    # Left subplot: annotated image with translucent bounding boxes
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
    plt.title("Clustered bounding boxes (translucent fill)")
    plt.axis("off")
    
    # Right subplot: original image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis("off")
    
    plt.tight_layout()

    # Save the result as result.png
    plt.savefig("result.png", dpi=300)  

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--kernel", type=int, default=15, help="Kernel size for morphological operations")
    parser.add_argument("--limit_cluster", type=bool, default=False, help="Limit the number of clusters to 4")
    parser.add_argument("--iteration",type=int,default=2,help="Number of iterations for morphological operations")
    
    args=parser.parse_args()

    # 1. Get bounding boxes from your image

    image_path = args.image


    image = cv2.imread(image_path)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_val = np.mean(gray)

    # If the image is dark, invert it
    if mean_val < 128:
        gray = cv2.bitwise_not(gray)
    else:
        gray = gray.copy()


    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.kernel,args.kernel))
    dilate = cv2.dilate(thresh, kernel, iterations=args.iteration)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    bboxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append((x, y, w, h))

    print(f"Found {len(bboxes)} bounding boxes.")



    # (Already done above, resulting in `bboxes`)

    # 2. Cluster using HDBSCAN
    
    if len(bboxes) < 2:
        # Not enough bounding boxes for clustering; assign noise label
        labels = np.full(len(bboxes), -1)
    else:
        labels = cluster_bounding_boxes_hdbscan(bboxes, min_cluster_size=2, min_samples=1)


    
    # 3. (Optional) Merge clusters if more than 4
    if args.limit_cluster:
        final_labels = limit_clusters(labels, max_clusters=4, bboxes=bboxes)
    else:
        final_labels = labels


    # Let’s see how many clusters we ended up with
    unique_final = [c for c in np.unique(final_labels) if c != -1]  # otherwise put final_labels
    print("Cluster labels (excluding noise):", unique_final)
    print("Number of clusters (excluding noise):", len(unique_final))

    visualize_clusters_translucent(original, bboxes, final_labels)

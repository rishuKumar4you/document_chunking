# Bounding Box Clustering with HDBSCAN

This Python script performs bounding box detection, clustering using HDBSCAN, and visualization on an input image. It's designed to group detected objects (represented by bounding boxes) into meaningful clusters.

## Table of Contents

* [Prerequisites](/#prerequisites)  
* [Installation](/#installation)  
* [Usage](/#usage)  
* [Code Explanation](/#code-explanation)  


## Prerequisites

* Python 3.x  
* OpenCV (\`cv2\`)  
* NumPy (\`numpy\`)  
* Matplotlib (\`matplotlib\`)  
* scikit-learn (\`sklearn\`)  
* HDBSCAN (\`hdbscan\`)

You can install these dependencies using pip:

```bash  
pip install opencv-python numpy matplotlib scikit-learn hdbscan 
```

## **Installation**

1. **Clone the repository:**  
   ```bash
    git clone https://github.com/rishuKumar4you/document_chunking  
    cd layout_analysis
   ```

## **Usage**

1. **Run the script:**  
  ``` python main.py --image <path_to_your_image> [arguments]```

   * --image: (Required) Path to the input image file.  
   * --kernel: (Optional, default: 15) Kernel size for morphological operations (dilation). Larger values merge nearby contours.  
   * --limit_cluster: (Optional, default: False) If set to True, the script will attempt to limit the number of clusters to 4 by merging smaller clusters.

**Example:** 
```bash
python main.py --image image.png --kernel 5 --limit_cluster True
```

2. **Output:**  
   * The script will display a plot with the original image and the clustered bounding boxes (with translucent fills).  
   * It will also save the visualization to a file named result.png in the same directory.  
   * The terminal will print the number of bounding boxes found and the number of clusters identified (excluding noise).

## **Code Explanation**

### **1\. Bounding Box Detection**

The script uses OpenCV to perform the following steps:

1. **Read the image:** cv2.imread() loads the input image.  
2. **Convert to grayscale:** cv2.cvtColor() converts the image to grayscale for thresholding.  
3. **Handle dark images:** If the average pixel value is low, the image is inverted using cv2.bitwise\_not() to ensure proper thresholding.  
4. **Gaussian blur:** cv2.GaussianBlur() smooths the image to reduce noise.  
5. **Otsu's thresholding:** cv2.threshold() applies Otsu's method to automatically determine the threshold value for binary conversion.  
6. **Dilation:** cv2.dilate() expands the white regions in the binary image using a rectangular kernel. This helps to merge nearby contours.  
7. **Contour detection:** cv2.findContours() finds the contours (object boundaries) in the dilated image.  
8. **Bounding box extraction:** cv2.boundingRect() extracts the bounding boxes (x, y, width, height) for each contour.

### **2. HDBSCAN Clustering**

The script uses HDBSCAN to cluster the detected bounding boxes:


1. **HDBSCAN fitting:** hdbscan.HDBSCAN() clusters the feature vectors. The min\_cluster\_size and min\_samples parameters can be adjusted to control the clustering behavior.  
2. **Cluster label assignment:** Each bounding box is assigned a cluster label based on the HDBSCAN results.  
3. **Cluster limiting (optional):** The limit\_clusters() function merges smaller clusters until the number of clusters is less than or equal to 4\. Clusters are merged based on proximity of their centroids.


## **Customization**

* **Kernel size:** Adjust the \--kernel argument to control the dilation effect.  
* **HDBSCAN parameters:** Modify the min\_cluster\_size and min\_samples parameters in the cluster\_bounding\_boxes\_hdbscan() function to fine-tune the clustering.  


## 

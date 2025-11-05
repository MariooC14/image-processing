# 0) One-time setup (data + imports) 
1. Extract dataset
   * Unzip /content/att_faces.zip to /content/att_faces. 
   * Directory layout should be att_faces/s1/1.pgm ... att_faces/s40/10.pgm. 

2. Set parameters 
   * DATA_DIR='att_faces', NUM_SUBJECTS=40, TRAIN_PER_SUBJECT=9, IMAGE_SIZE=(112,92), K=200, etc. 
   * Optionally set NEW_TEST_IMG_PATH to try an external face image later. 

3. Choose saved vs fresh build 
   * USE_SAVED_DATA=False (first run) builds and saves A and mean_image to face_data_eigenfaces_orl.npz. 
   * Subsequent runs: set USE_SAVED_DATA=True for speed. 


# 1) Build the training matrix $A$

**What**: Read the first 9 images for each subject, flatten to column vectors, and stack them.

**Why**: PCA expects a data matrix with one sample per column.

* For each subject s1 ... s40, read images 1.pgm ... 9.pgm. 
* Flatten each image of size 112×92 to a vector of length 10304. 
* Place each vector as a column in $A$ → shape (10304,360).

# 2) Mean-center the data
**What**: Compute the mean face (per-pixel mean across all training images) and subtract it from every column of $A$.

**Why**: PCA requires zero-mean data.

* `mean_image = mean(A, axis=1)` → vector of length 10304.
* `A_centered = A - mean_image[:, None]` (the code stores this back into A)

# 3) “Small covariance” trick for PCA
**What**: Instead of eigen-decomposing $L=AA^T$ (huge 10304×10304), use $C=A^TA$ (small 360×360).

**Why**: Much faster when #pixels ≫ #images. 

* Compute $C=A^TA$. 
* Eigendecompose $C$ (descending order of eigenvalues).

# 4) Compute Eigenfaces
**What**: Map eigenvectors of $C$ into pixel space to get eigenvectors of $L$ (the eigenfaces).

**How**:  $vectorL_{all}=A \cdot eigvecs_C$.

**Normalize**: Make each eigenface unit-norm (column-wise).

**Select**: Keep top K eigenfaces → `vectorL = vectorL_all[:, :K]`.

# 5) Project training images (get coefficients) 

What: Represent each training face as coordinates in the “face space”. 
How: $coeff_{all}=A^⊤ \cdot vectorL_{all}$ , and then take first K columns → coeff.

# 6) Build a subject model (class prototypes)
**What**: For each subject, average their 9 training coefficients to a single centroid.

**Why**: Classification = nearest centroid in coefficient space.

Result: model has shape(40,K).

# 7) Test on held-out images
What: For each subject, use their 10th image as test.

Steps: 
1. Read test image, flatten, mean-center with mean_image.
2. Project to face space: $test_coeff=x⊤vectorL$.
3. Compute distances to all 40 centroids in model.
4. Predicted subject = nearest centroid (smallest Euclidean distance). 
5. Report accuracy across all 40 test images.

# 8) Visualizations
1. Matrix A tiled: Show all mean-centered training images arranged by subject × image index.
2. Mean face: Display mean_image reshaped to 112×92.
3. Reconstructions (intuition for K): 
   * Pick a training image (by GIVEN_INDEX). 
   * Reconstruct with all eigenfaces vs top K eigenfaces. 
   * Compare quality to understand dimensionality vs fidelity. 


# 9) External image test (not in training set) 
**What**: Try any face image to see which subject it resembles most.

Steps:
* Load external image (NEW_TEST_IMG_PATH), convert to grayscale, resize to 112×92, flatten.
* Mean-center, project to face space, compute nearest subject centroid. 
* Display the external image and the best-match subject’s sample (e.g., sX/1.pgm). 
# === Eigenfaces (ORL/AT&T dataset) — Colab-ready Python ===
# Mirrors the logic of your MATLAB script
# ----------------------------------------------------------
# Requirements (already available in Colab): numpy, matplotlib, imageio, pillow, scikit-image
# If needed: !pip install imageio scikit-image

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

# ----------------------- Parameters -----------------------
DATA_DIR = 'att_faces'      # folder with s1 ... s40 subfolders, each with 1..10.pgm
NUM_SUBJECTS = 40
TRAIN_PER_SUBJECT = 9       # 9 train, 1 test (the 10th)
TEST_PER_SUBJECT = 1
IMAGE_SIZE = (112, 92)      # (H, W)
K = 200                     # number of eigenfaces to use
GIVEN_INDEX = 3             # an example index for reconstruction (1-based in MATLAB)
USE_SAVED_DATA = True       # Set False to rebuild A and mean face

SAVE_FILE = 'face_data_eigenfaces_orl.npz'  # where A and mean_image are saved

# Optional: new test image not in training set (set to None to skip)
# e.g., upload 'face.jpg' to /content and set NEW_TEST_IMG_PATH = '/content/face.jpg'
NEW_TEST_IMG_PATH = None

# ----------------------- Utilities ------------------------
def read_pgm(path):
    """Read .pgm as float64 vectorized (H*W,)."""
    img = imageio.imread(path)
    # imageio loads greyscale as (H, W); ensure dtype float64
    return img.astype(np.float64).reshape(-1)

def load_training_matrix(data_dir, num_subjects, train_per_subject, image_size):
    """Build A (pixels x num_train) from ORL/AT&T training images."""
    H, W = image_size
    num_train = num_subjects * train_per_subject
    A = np.zeros((H*W, num_train), dtype=np.float64)
    idx = 0
    for subject in range(1, num_subjects + 1):
        subject_dir = os.path.join(data_dir, f's{subject}')
        for img_num in range(1, train_per_subject + 1):
            img_path = os.path.join(subject_dir, f'{img_num}.pgm')
            A[:, idx] = read_pgm(img_path)
            idx += 1
    return A

def ensure_K(K, max_dim):
    return int(max(1, min(K, max_dim)))

def tile_matrix_A(A, image_size, num_subjects, train_per_subject):
    """Visualize A similar to MATLAB reshape trick, but using a clean tiling."""
    H, W = image_size
    # A has columns = each image. We'll tile by subjects (rows) x train_per_subject (cols).
    tiles = []
    col = 0
    for r in range(num_subjects):
        row_imgs = []
        for c in range(train_per_subject):
            img = A[:, col].reshape(H, W)
            row_imgs.append(img)
            col += 1
        tiles.append(np.hstack(row_imgs))
    return np.vstack(tiles)

def normalize_columns(M, eps=1e-12):
    norms = np.linalg.norm(M, axis=0) + eps
    return M / norms

# ----------------------- Load / Build ----------------------
if USE_SAVED_DATA and os.path.exists(SAVE_FILE):
    data = np.load(SAVE_FILE)
    A = data['A']
    mean_image = data['mean_image']
else:
    A = load_training_matrix(DATA_DIR, NUM_SUBJECTS, TRAIN_PER_SUBJECT, IMAGE_SIZE)
    mean_image = A.mean(axis=0)  # WRONG! (kept for clarity below)

    # Fix: Mean per pixel across columns
    mean_image = A.mean(axis=1, keepdims=False)  # (pixels,)
    A = A.astype(np.float64) - mean_image[:, None]
    np.savez_compressed(SAVE_FILE, A=A, mean_image=mean_image)

# ----------------------- PCA via C = A'A -------------------
# For explanation: the big covariance would be L = A A^T, but we use C = A^T A (smaller).
C = A.T @ A  # shape (num_train, num_train)
# Symmetric => use eigh. Sort descending.
eigvals, eigvecs_C = np.linalg.eigh(C)
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs_C = eigvecs_C[:, order]

# Eigenfaces (in pixel space) are columns of A @ eigvecs_C
vectorL_all = A @ eigvecs_C          # (pixels, num_train)
vectorL_all = normalize_columns(vectorL_all)

# Use top-K eigenfaces
K = ensure_K(K, vectorL_all.shape[1])
vectorL = vectorL_all[:, :K]

# ----------------------- Projection onto Face space-------------------
# Training coefficients: project each training image (columns of A) into face space
# In MATLAB: coeff_all = A' * vectorL_all  -> shape (num_train, num_train)
coeff_all = (A.T @ vectorL_all)              # (num_train, num_train)
coeff = coeff_all[:, :K]                     # (num_train, K)

# ----------------------- Subject Models --------------------
# Mean coefficient vector per subject (across their TRAIN_PER_SUBJECT images)
model = np.zeros((NUM_SUBJECTS, K), dtype=np.float64)
for i in range(NUM_SUBJECTS):
    start = i * TRAIN_PER_SUBJECT
    end = (i + 1) * TRAIN_PER_SUBJECT
    model[i, :] = coeff[start:end, :].mean(axis=0)

# ----------------------- Testing ---------------------------
# 1 test image per subject: the (TRAIN_PER_SUBJECT + 1)-th image in each subject folder
correct = 0
num_test_images = NUM_SUBJECTS * TEST_PER_SUBJECT

for subject in range(1, NUM_SUBJECTS + 1):
    subject_dir = os.path.join(DATA_DIR, f's{subject}')
    test_img_path = os.path.join(subject_dir, f'{TRAIN_PER_SUBJECT + 1}.pgm')
    test_img = imageio.imread(test_img_path).astype(np.float64).reshape(-1)
    test_img_centered = test_img - mean_image
    test_coeff = (test_img_centered @ vectorL)    # shape (K,)

    # Find nearest subject centroid in coefficient space
    dists = np.linalg.norm(model - test_coeff[None, :], axis=1)
    predicted_subject = 1 + np.argmin(dists)

    if predicted_subject == subject:
        correct += 1

accuracy = 100.0 * correct / num_test_images
print(f"Test Accuracy: {accuracy:.2f}%")

# ----------------------- Visualizations --------------------
# A (mean-centered) tiled
A_tiled = tile_matrix_A(A, IMAGE_SIZE, NUM_SUBJECTS, TRAIN_PER_SUBJECT)
plt.figure(figsize=(8, 10))
plt.imshow(A_tiled, cmap='gray')
plt.title('Full Matrix A (mean-centered) as an Image')
plt.axis('off')
plt.show()

# Mean face
plt.figure(figsize=(4, 5))
plt.imshow(mean_image.reshape(IMAGE_SIZE), cmap='gray')
plt.title('Mean Face')
plt.axis('off')
plt.show()

# ----------------------- Reconstructions -------------------
# Convert GIVEN_INDEX to 0-based index into training set
# The given index refers to a training image column in A
gi = max(0, min(A.shape[1] - 1, GIVEN_INDEX - 1))
original_face = (A[:, gi] + mean_image).reshape(IMAGE_SIZE)

# Reconstruct using all eigenfaces
coeff_all_example = coeff_all[gi, :]                 # shape (num_train,)
reconstructed_all = mean_image + (vectorL_all @ coeff_all_example).reshape(-1)

# Reconstruct using top-K eigenfaces
coeff_K_example = coeff[gi, :]                       # shape (K,)
reconstructed_K = mean_image + (vectorL @ coeff_K_example).reshape(-1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_face, cmap='gray')
plt.title(f'{GIVEN_INDEX} Given Train Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_all.reshape(IMAGE_SIZE), cmap='gray')
plt.title('Reconstructed with All Eigenfaces')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_K.reshape(IMAGE_SIZE), cmap='gray')
plt.title(f'Reconstructed with {K} Eigenfaces')
plt.axis('off')
plt.show()

# ----------------- Test image not in training set ----------
if NEW_TEST_IMG_PATH is not None and os.path.exists(NEW_TEST_IMG_PATH):
    new_img = imageio.imread(NEW_TEST_IMG_PATH)
    # Convert to grayscale if RGB
    if new_img.ndim == 3:
        new_img = (rgb2gray(new_img) * 255.0).astype(np.float64)
    # Resize if needed
    if new_img.shape != IMAGE_SIZE:
        new_img = resize(new_img, IMAGE_SIZE, anti_aliasing=True)
        new_img = (new_img * 255.0).astype(np.float64)

    new_vec = new_img.reshape(-1).astype(np.float64)
    new_vec_centered = new_vec - mean_image
    new_coeff = (new_vec_centered @ vectorL)  # (K,)

    dists = np.linalg.norm(model - new_coeff[None, :], axis=1)
    predicted_subject = 1 + np.argmin(dists)

    # Load a representative image (e.g., 1.pgm) of the predicted subject
    closest_img_path = os.path.join(DATA_DIR, f's{predicted_subject}', '1.pgm')
    closest_img = imageio.imread(closest_img_path)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(new_img, cmap='gray')
    plt.title('Original External Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(closest_img, cmap='gray')
    plt.title(f'Best Match: s{predicted_subject}')
    plt.axis('off')
    plt.show()

# ----------------------- Credits ---------------------------
print("\nPlease credit the Olivetti Research Laboratory for the ORL faces dataset.")
print('Reference: F. Samaria and A. Harter, "Parameterisation of a stochastic model for human face identification"')
print('2nd IEEE Workshop on Applications of Computer Vision, Dec 1994, Sarasota (Florida).')

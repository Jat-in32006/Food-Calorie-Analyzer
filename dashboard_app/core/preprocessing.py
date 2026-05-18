import cv2
import numpy as np

def preprocess_food_image(image_bgr: np.ndarray) -> dict:
    """
    Applies Unit 6 (SVD), Unit 1 (HSV), and Unit 2 (CLAHE & Dual Spatial Filters)
    to isolate structural food items independent of background patterns and lighting variance.
    """
    # 1. Unit 6: Background Noise Reduction using Singular Value Decomposition (SVD)
    # Convert image to single-channel float to perform decomposition
    gray_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    U, Sigma, Vt = np.linalg.svd(gray_raw, full_matrices=False)
    
    # Truncate lower singular values that represent high-frequency noise/tablecloth patterns
    # Retain the top 85% dominant energy vectors
    k = int(len(Sigma) * 0.85)
    Sigma_truncated = np.zeros((Sigma.shape[0], Sigma.shape[0]))
    for i in range(k):
        Sigma_truncated[i, i] = Sigma[i]
    
    reconstructed_gray = np.dot(U, np.dot(Sigma_truncated, Vt))
    gray_clean = cv2.normalize(reconstructed_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Unit 1: Color Model Transformation to HSV
    # Decouple color/chroma components from lighting/luminance variances
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    
    # 3. Unit 2: Local Contrast Enhancement via CLAHE
    # Equalize local contrast on the Value (V) channel to balance harsh flashes or dim lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)
    
    # Reconstruct the color image using the illumination-balanced channel
    enhanced_hsv = cv2.merge([h, s, v_equalized])
    enhanced_bgr = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    # 4. Unit 2: Dual Spatial Filtering for Edge Preservation & Glare Reduction
    # Gaussian blur handles camera sensor noise; Median blur strips specular glare/reflections
    blurred_spatial = cv2.GaussianBlur(gray_clean, (5, 5), 0)
    filtered_final = cv2.medianBlur(blurred_spatial, 5)
    
    return {
        "processed_gray": filtered_final,
        "enhanced_bgr": enhanced_bgr,
        "hsv_channels": (h, s, v_equalized)
    }

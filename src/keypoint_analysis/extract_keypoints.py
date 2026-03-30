import cv2

def extract_keypoints(image, nfeatures=5000):
    """
    Extract ORB keypoints from an image.

    Args:
        image (np.array): input image (BGR)
        nfeatures (int): max number of keypoints

    Returns:
        keypoints (list): detected keypoints
        keypoint_count (int): number of keypoints
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints = orb.detect(gray, None)

    return keypoints, len(keypoints)

def draw_keypoints(image, keypoints):
    """
    Draw keypoints on image for visualization.
    """
    return cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT
    )
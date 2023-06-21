import cv2


def detect_gender(face_image):
    """
    Detects gender in the given face image.

    Args:
        face_image (ndarray): The face image.

    Returns:
        str: 'Male' or 'Female'.
    """

    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt",
        "gender_net.caffemodel"
    )

    blob = cv2.dnn.blobFromImage(
        face_image, 1.0,
        (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False,
        crop=False
    )

    gender_net.setInput(blob)
    gender_pred = gender_net.forward()
    gender = "Male" if gender_pred[0][0] > 0.5 else "Female"

    return gender


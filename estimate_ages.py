import cv2
import numpy as np
import detect_persons
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='resnet50', include_top=False, pooling='avg')


def estimate_ages(frame):
    persons = detect_persons.detect_persons(frame)
    ages = []

    for x, y, w, h in persons:
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))

        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face, version=2)

        predictions = model.predict(face)

        age = predictions[0]

        ages.append(age)

    return ages

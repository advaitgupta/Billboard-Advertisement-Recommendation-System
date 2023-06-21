import cv2
import numpy as np


def detect_persons(frame):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    detections = net.forward(output_layers)

    persons = []
    boxes= []

    height, width = frame.shape[:2]

    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if class_id == 0:
                    persons.append((x, y, w, h))
                    boxes.append([x, y, w, h])

    return persons

#
# 
# def draw_bounding_boxes(frame, boxes):
#     for x, y, w, h in boxes:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     return frame
#
#
# def save_image(image_path, image):
#     cv2.imwrite(image_path, image)
#     print(f"Image saved at {image_path}")
#
#
# image_path = '/Users/advaitgupta/Downloads/22033.jpg'
# output_image_path = '/Users/advaitgupta/Downloads/result2.jpg'
#
# frame = cv2.imread(image_path)
# persons, boxes = detect_persons(frame)
# result_frame = draw_bounding_boxes(frame.copy(), boxes)
# save_image(output_image_path, result_frame)
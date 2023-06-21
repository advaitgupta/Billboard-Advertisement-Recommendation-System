import cv2
import numpy as np


def load_yolo(config_path, weights_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


def detect_vehicles(frame, config_path, weights_path):
    net, output_layers = load_yolo(config_path, weights_path)
    detected_vehicles = []
    if frame is None:
        print("None")
        return detected_vehicles

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    vehicle_classes = [2, 7, 14]  # 2 for car, 7 for truck and 14 for motorbike
    detected_vehicles = [class_id for class_id in class_ids if class_id in vehicle_classes]

    return detected_vehicles

# def draw_bounding_boxes(frame, boxes, class_ids):
#     colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
#
#     for i, (x, y, w, h) in enumerate(boxes):
#         label = f"Vehicle {class_ids[i]}"
#         print("label")
#         # label = "Vehicle"
#         color = colors[i]
#         # color = 2
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return frame

# def save_image(image_path, image):
#     cv2.imwrite(image_path, image)
#     print(f"Image saved at {image_path}")


# config_path = "yolov3.cfg"
# weights_path = "yolov3.weights"
# image_path1 = '/Users/advaitgupta/Downloads/profile1662380816.jpg'
# frame = cv2.imread(image_path1)
#
# detected_vehicles = detect_vehicles(frame, config_path, weights_path)
# print("boxes length")
# print(len(boxes))
# print(boxes)
# print("detected vehicle")
# print(detected_vehicles)
# result_frame = draw_bounding_boxes(frame.copy(), boxes, detected_vehicles)
#
# output_image_path = '/Users/advaitgupta/Downloads/result_image.jpg'
# save_image(output_image_path, result_frame)

from detect_vehicles import detect_vehicles
from detect_persons import detect_persons
from classify_human_groups import classify_human_groups
from recommend_ad_collaborative_based import recommend_ad
from recommend_ad_content_based import recommend_ad as recommend_ad1
from classify_vehicles import classify_vehicles
import cv2


def main(image_path):
    frame = cv2.imread(image_path)

    # Detecting the vehicles
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    vehicles = detect_vehicles(frame, config_path, weights_path)
    vehicle_groups = classify_vehicles(vehicles)

    # Detecting persons and classifying them among the human groups
    human_groups = classify_human_groups(frame)

    # Use recommend_ad for Collaborative Filtering System and recommend_ad1 for Content Based
    ad = recommend_ad1(human_groups, vehicle_groups)

    print(human_groups)
    print(vehicle_groups)

    print(f"Recommended Advertisement: {ad}")


if __name__ == "__main__":
    image_path1 = '/Users/advaitgupta/Desktop/Projects & Study/Projects/Personal/Billboard Advertisement Recommendation System/22033.jpg'
    main(image_path1)


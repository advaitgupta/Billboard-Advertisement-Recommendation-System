import detect_persons
from analyze_spatial_relationships import analyze_spatial_relationships


def compute_groups(frame, output_image_path="/Users/advaitgupta/Downloads/output5.jpg"):
    """
    Compute groups based on spatial proximity, draw bounding boxes around each group, and save the resultant image.
    Args:
        frame : image frame.
        output_image_path: path to save the output image.

    Returns:
        list: List of groups [[person1, person2, ...], ...].
    """
    persons = detect_persons.detect_persons(frame)

    def center(box):
        x, y, w, h = box
        return x + w / 2, y + h / 2

    positions = [center(box) for box in persons]

    labels = analyze_spatial_relationships(positions, eps=40, min_samples=1)

    groups = [[] for _ in range(max(labels) + 1)]
    for person, label in zip(persons, labels):
        # print(label)
        if label != -1:  # -1 label is for noise points, not assigned to any cluster
            groups[label].append(person)

    # for person in persons:
    #     x, y, w, h = person
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate and display age
        # face = frame[y:y + h, x:x + w]
        # age = estimate_ages.estimate_ages(face)
        # age_text = f"{age}" if age else "Unknown"
        # cv2.putText(frame, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw bounding boxes around each group
    # for group in groups:
    #     if group:
    #         x_min = min([x for x, y, w, h in group])
    #         y_min = min([y for x, y, w, h in group])
    #         x_max = max([x + w for x, y, w, h in group])
    #         y_max = max([y + h for x, y, w, h in group])
    #         face = frame[y_min:y_max, x_min:x_max]
    #
    #         # Draw rectangle around group
    #         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #
    #         group_length = len(group)
    #         cv2.putText(frame, f"{group_length}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
    #                     (0, 255, 0), 2)

    # Save the modified frame with bounding boxes
    # cv2.imwrite(output_image_path, frame)

    return groups

# image_path = '/Users/advaitgupta/Downloads/22034.jpg'
# frame = cv2.imread(image_path)
# compute_groups(frame)

import numpy as np
import estimate_ages
import cv2


def identify_groups_of_children(frame, groups):
    groups_of_children = []
    for group in groups:
        if group:
            x_min = min([x for x, y, w, h in group])
            y_min = min([y for x, y, w, h in group])
            x_max = max([x + w for x, y, w, h in group])
            y_max = max([y + h for x, y, w, h in group])
            frame1 = frame[y_min:y_max, x_min:x_max]
            ages = estimate_ages.estimate_ages(frame1)
            if ages:
                flat_ages = np.concatenate(ages)  # Flatten the list of lists into a single array
                if np.all(flat_ages < 18) and len(group) > 1:
                    groups_of_children.append(group)

    return groups_of_children

import numpy as np
import estimate_ages
import estimate_sizes
import cv2


def identify_groups_with_children(frame, groups):
    age_variation = 0
    size_variation = 0

    families = []
    for group in groups:
        if group:
            x_min = min([x for x, y, w, h in group])
            y_min = min([y for x, y, w, h in group])
            x_max = max([x + w for x, y, w, h in group])
            y_max = max([y + h for x, y, w, h in group])
            frame1 = frame[y_min:y_max, x_min:x_max]
            ages = estimate_ages.estimate_ages(frame1)
            sizes = estimate_sizes.estimate_sizes(frame1)
            if ages:
                age_variation = np.amax(ages) - np.amin(ages)
            if sizes:
                size_variation = max(sizes) - min(sizes)

            if age_variation > 20 and size_variation > 200 and len(group) > 2:
                families.append(group)

    return families

import detect_gender


def identify_singles(frame, groups):
    single_men = []
    single_women = []
    for group in groups:
        if len(group) == 1:
            x_min = min([x for x, y, w, h in group])
            y_min = min([y for x, y, w, h in group])
            x_max = max([x + w for x, y, w, h in group])
            y_max = max([y + h for x, y, w, h in group])
            frame1 = frame[y_min:y_max, x_min:x_max]
            gender = detect_gender.detect_gender(frame1)

            if gender == "Male":
                single_men.append(group)
            elif gender == "Female":
                single_women.append(group)
    return single_men, single_women

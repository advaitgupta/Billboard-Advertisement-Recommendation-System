import detect_persons


def estimate_sizes(frame):
    persons = detect_persons.detect_persons(frame)
    sizes = []
    for person in persons:
        x, y, w, h = person
        size = w * h
        sizes.append(size)
    return sizes

import numpy as np
import identify_groups_of_children
import identify_groups_with_children


def identify_groups_of_people(frame, groups):
    people = []

    for group in groups:
        children = identify_groups_of_children.identify_groups_of_children(frame, [group])
        family = identify_groups_with_children.identify_groups_with_children(frame, [group])

        if len(group) > 2 and not children and not family:
            people.append(group)

    return people

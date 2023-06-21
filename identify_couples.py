import identify_groups_of_children
import copy


def identify_couples(frame, groups):
    couples = []
    groups_copy = copy.deepcopy(groups)

    for group in groups_copy:
        children = identify_groups_of_children.identify_groups_of_children(frame, [group])
        if len(group) == 2 and not children:
            couples.append(group)
    return couples

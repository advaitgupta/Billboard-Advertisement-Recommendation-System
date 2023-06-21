import identify_groups_with_children
import identify_singles
import identify_groups_of_children
import identify_couples
import compute_groups
import detect_persons
import estimate_sizes
import estimate_ages
import identify_groups_of_people


def classify_human_groups(frame):

    groups = compute_groups.compute_groups(frame)

    families = identify_groups_with_children.identify_groups_with_children(frame, groups)
    groups_of_children = identify_groups_of_children.identify_groups_of_children(frame, groups)
    couples = identify_couples.identify_couples(frame, groups)
    single_men, single_women = identify_singles.identify_singles(frame, groups)
    people = identify_groups_of_people.identify_groups_of_people(frame, groups)

    human_groups = {
        'families': len(families),
        'groups_of_children': len(groups_of_children),
        'couples': len(couples),
        'single_men': len(single_men),
        'single_women': len(single_women),
        'groups_of_people': len(people)
    }

    return human_groups

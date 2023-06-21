# recommend_ad_collaborative_based.py

import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def normalize_counts(counts):
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total > 0 else counts


def recommend_ad(human_groups, vehicle_types):
    ad_preferences = joblib.load('ad_preferences.pkl')
    combined_groups = {**human_groups, **vehicle_types}

    categories = ['families', 'groups_of_children', 'couples', 'single_men', 'single_women', 'car', 'truck', 'bike']

    similarity = cosine_similarity(ad_preferences)

    normalized_combined = normalize_counts(combined_groups)

    new_normalized_combined = np.array([normalized_combined.get(group, 0) for group in categories])

    weighted_preferences = new_normalized_combined @ similarity

    recommended_ad = np.argmax(weighted_preferences @ ad_preferences) + 1

    return f"Ad {recommended_ad}"


# human_groups1 = {
#     'families': 0,
#     'groups_of_children': 0,
#     'couples': 0,
#     'single_men': 0,
#     'single_women': 0
# }
#
# vehicle_types1 = {
#     'car': 0,
#     'truck': 50,
#     'bike': 0
# }
#
# recommendation = recommend_ad(human_groups1, vehicle_types1)
# print(recommendation)







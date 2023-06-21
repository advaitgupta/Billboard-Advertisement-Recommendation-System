import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def normalize_counts(counts):
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total > 0 else counts


def recommend_ad(human_groups, vehicle_types):
    ad_preferences = joblib.load('ad_preferences.pkl')
    categories = ['families', 'groups_of_children', 'couples', 'single_men', 'single_women', 'groups_of_people', 'car', 'truck', 'bike']

    normalized_combined = normalize_counts({**human_groups, **vehicle_types})

    new_normalized_combined = np.array([normalized_combined.get(category, 0) for category in categories]).reshape(1, -1)

    # print(new_normalized_combined.shape)
    # print(ad_preferences.T.shape)

    # Computing cosine similarity between new_normalized_combined and ad_preferences
    similarity = cosine_similarity(new_normalized_combined, ad_preferences.T)

    print(similarity)

    recommended_ad_category = np.argmax(similarity) + 1

    return f"Ad {recommended_ad_category}"


# human_groups1 = {
#     'families': 0,
#     'groups_of_children': 0,
#     'couples': 0,
#     'single_men': 0,
#     'single_women': 5
# }
#
# vehicle_types1 = {
#     'car': 0,
#     'truck': 2,
#     'bike': 10
# }
#
# recommendation = recommend_ad(human_groups1, vehicle_types1)
# print(recommendation)

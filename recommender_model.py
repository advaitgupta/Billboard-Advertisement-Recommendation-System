import pandas as pd
import numpy as np
import joblib

# data = pd.read_csv('historical_ad_preferences.csv')

data = {
    'Category': ['families', 'groups_of_children', 'couples', 'single_men', 'single_women', 'groups_of_people', 'car', 'truck', 'bike'],
    'Ad 1': [2, 1, 4, 5, 4, 3, 2, 3, 7],
    'Ad 2': [0, 4, 2, 1, 3, 3, 7, 3, 9],
    'Ad 3': [3, 5, 4, 4, 3, 3, 3, 4, 10]
}

df = pd.DataFrame(data)

ad_preferences = df.iloc[:, 1:].values

joblib.dump(ad_preferences, 'ad_preferences.pkl')

print(ad_preferences)
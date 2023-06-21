from sklearn.cluster import DBSCAN


def analyze_spatial_relationships(positions, eps=3, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
    return clustering.labels_

def classify_vehicles(detected_vehicles):
    category_mapping = {
        2: "Car",
        7: "Truck",
        14: "Bike"
    }

    vehicle_counts = {}
    for vehicle in detected_vehicles:
        category = category_mapping.get(vehicle)
        vehicle_counts[category] = vehicle_counts.get(category, 0) + 1

    return vehicle_counts

# Billboard Advertisement Recommendation System

The Billboard Advertisement Recommendation System is an intelligent and dynamic advertising platform that selects advertisements to be displayed on billboards based on real-time traffic and pedestrian demographics. The system employs computer vision, machine learning algorithms, and content-based filtering to analyze and understand the audience present at a given time and then display the most relevant advertisements. The dataset is still not fully available and we are in the process of collecting data from various sources. We are using a hard-coded temporary dataset for the time being.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dependencies](#dependencies)

## Features

- Utilizes computer vision algorithms with the YOLO (You Only Look Once) model to detect and count vehicles and pedestrians from traffic camera feeds in real-time.
- Employs face analysis to estimate the age and identify the gender of pedestrians by integrating VGGFace with ResNet50.
- Detect various groups of people such as families, groups of children, couples, single men and women and groups of people using computer vision.
- Uses clustering algorithms, specifically DBSCAN, to analyze spatial relationships among pedestrians and categorize them into groups such as families, groups of children, couples, single men, and single women.
- Groups are formed based on age, size, and spatial proximity.
- Classify vehicles into categories such as cars, trucks, and bikes.
- Recommend an advertisement based on the groups of people and types of vehicles detected.
- Content-based filtering is implemented using cosine similarity to calculate the relevance of advertisements based on the demographic composition and vehicle types.
- The system computes a weighted sum of advertisement preferences based on the cosine similarity of the current scene to historical advertisement performance data.
- Advertisements with the highest weighted preference are selected for display.
- The system can also employ collaborative filtering as an alternative method for advertisement recommendations.
- Traffic data is normalized to calculate the percentage representation of different demographic groups and vehicle types in the current scene.
- This allows for more accurate recommendations based on the relative significance of different audience segments.
- Works with images.

## Installation

1. Clone this repository:
git clone https://github.com/yourusername/BillboardAdvertisementRecommendationSystem.git


2. Navigate to the cloned repository:
cd BillboardAdvertisementRecommendationSystem


3. Install the required dependencies


## Usage

1. Add the input image you want to analyze to the project directory.

2. Run the main script (Make sure to change the location of image file in main function)

## Dependencies

This project is built using Python and relies on several libraries including:

- OpenCV
- NumPy
- scikit-learn
- Keras VGGFace
- Joblib
- Yolov3

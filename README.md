# CFS_Python
Collaborative Filtering System in Python 

A modular, Python-based movie recommendation system built with collaborative filtering techniques using the MovieLens 100K dataset.

# Project Overview
This project implements a complete recommendation pipeline from data loading to personalized movie suggestions. It features multiple collaborative filtering algorithms with performance comparison capabilities.

# System Architecture
Core Modules
DataLoader - Handles dataset loading and initial structuring

DataCleaner - Manages data preprocessing and utility matrix creation

SimilarityCalculator - Computes user-user and item-item similarity matrices

MovieRecommender - Implements prediction algorithms and recommendation generation

ML100KRecommendationSystem - Main orchestrator with interactive interface

Supported Algorithms
User-Based Collaborative Filtering - Recommendations based on similar users

Item-Based Collaborative Filtering - Recommendations based on similar movies

Hybrid Approach - Combined user-based and item-based methods

# Dataset
The system uses the MovieLens 100K dataset containing:

100,000 ratings from 943 users on 1,682 movies

Rating scale: 1-5 stars

Movie metadata (genres, release dates)

User demographic information

# Installation & Setup
Prerequisites
bash
Python 3.7+
Install Dependencies
bash
pip install pandas numpy scikit-learn scipy
Download Dataset
Place the MovieLens 100K dataset files (u.data, u.item, u.user) in a dataset/ directory at the project root.

# Usage
Quick Start
python
from main import ML100KRecommendationSystem

# Initialize the system
system = ML100KRecommendationSystem(data_path="dataset")
system.initialize_system()

# Get recommendations for user 123
recommendations = system.get_recommendations(user_id=123, n_recommendations=5, method='hybrid')
Interactive Mode
bash
python main.py
Choose interactive mode to:

Get recommendations for specific users

Compare different algorithms

View most active users

Run performance tests

# Performance Metrics
The system evaluates recommendations using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Precision@k (Top-N recommendation quality)

Execution Time comparison

# Example Output

🎯 RECOMMENDATIONS FOR USER 123 (HYBRID)
============================================================

🌟 TOP 3 RECOMMANDATIONS:
------------------------------------------------------------
 1. The Shawshank Redemption (1994)
    📊 Score: 4.72 | Genres: Drama
    🗓️  Year: 1994

 2. Pulp Fiction (1994)
    📊 Score: 4.65 | Genres: Crime, Drama
    🗓️  Year: 1994

 3. The Godfather (1972)
    📊 Score: 4.61 | Genres: Crime, Drama
    🗓️  Year: 1972
# Project Structure

movie-recommendation-system/
├── data_loader.py          # Data loading module
├── data_cleaning.py        # Data preprocessing module
├── similarity.py           # Similarity calculations
├── recommender.py          # Recommendation algorithms
├── main.py                 # Main system and interactive interface
├── dataset/                # MovieLens 100K data files
│   ├── u.data
│   ├── u.item
│   └── u.user
└── results/                # Generated recommendations and logs
# Key Features
Modular Design: Clean separation of concerns for easy maintenance

Multiple Algorithms: Compare user-based, item-based, and hybrid approaches

Performance Evaluation: Built-in metrics for algorithm comparison

Interactive Interface: User-friendly CLI for exploration

Efficient Computation: Sparse matrix operations and optimized similarity calculations

Result Export: Save recommendations to CSV for analysis

# Technical Details
Similarity Calculation
Uses cosine similarity for user-user and item-item comparisons

Implements sparse matrix operations for memory efficiency

Optional dimensionality reduction with Truncated SVD

Missing Value Handling
Missing ratings represented as 0 in utility matrix

Similarity calculations ignore mutual zeros

Fallback strategies for cold-start scenarios

# Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add some AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Academic Use
This project serves as an excellent example for:

Machine Learning courses

Recommendation system studies

Collaborative filtering implementations

Python software architecture

# Contact
For questions or feedback, please open an issue on GitHub.

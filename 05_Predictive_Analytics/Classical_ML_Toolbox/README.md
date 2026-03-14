# Classical ML Toolbox

## Projects Overview
Machine Learning Scripts & Model Benchmarks
This repository contains various implementations focused on model comparison, data processing, and optimization. I used these projects to evaluate how different algorithms perform on real-world datasets.

01_Data_Preprocessing_Templates
Reusable templates for data cleaning. I used ColumnTransformer to handle numerical scaling and categorical encoding in one step, making the workflow efficient for new projects.

02_Predictive_Modeling_Regression_Benchmark
I compared five regression models—Linear, Polynomial, SVR, Decision Tree, and Random Forest—to predict salaries. I used StandardScaler for the SVR model and a 500-point grid to visualize the different regression curves.

03_Customer_Behavior_Classification_Comparison
I benchmarked classifiers like KNN and SVM for ad-click prediction. Results were evaluated using Confusion Matrices and ROC-AUC scores, with Decision Boundaries plotted to visualize class separation.

04_Customer_Segmentation (Clustering)
I applied K-Means and Hierarchical Clustering to group customer spending. I used the Elbow Method and Silhouette Scores to identify 5 optimal clusters for the dataset.

05_Market_Basket_Analysis
Implementation of the Apriori algorithm to find product associations. Rules were filtered by Support, Confidence, and Lift to extract the top 10 most significant buying patterns.

06_Thompson_Sampling_Optimization
A Reinforcement Learning project using Thompson Sampling for ad selection. It uses the Beta Distribution to balance exploration and exploitation, identifying the highest Click-Through-Rate in real-time.

07_Wine_Dimensionality_Comparison
I compared PCA, LDA, and Kernel PCA for dimensionality reduction. I used Logistic Regression to visualize which method created the best 2D separation between wine categories.

08_Hyperparameter_Tuning (grid_search & kfold)
Used Grid Search to optimize model parameters and K-Fold Cross Validation to ensure accuracy results were consistent across different data splits.
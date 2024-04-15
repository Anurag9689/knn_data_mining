# %%
# Importing necessary libraries
import numpy as np  # Required for efficient computations 
import pandas as pd  # Necessary for data handling
import seaborn as sns  # Utilized for statistical data visualization 
import matplotlib.pyplot as plt  # Used for plotting to visualize data


# %%
# Loading the dataset
# The dataset contains information related to diabetes patients.
# It includes various attributes such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.
# Each row represents a patient, and the last column ('Outcome') indicates whether the patient has diabetes (1) or not (0).
diabetes_df = pd.read_csv('diabetes.csv')
diabetes_df.head(10)

# %%
# Data visualization
# Pairplot visualization is utilized to explore the relationships between different variables in the dataset.
# It helps in understanding the distribution and correlation between variables.
sns.pairplot(diabetes_df, hue='Outcome', diag_kind='kde')
# plt.show()

# %%
# Implementing kNN Algorithm
# k-Nearest Neighbors (kNN) algorithm is implemented for classification.
# It calculates the distance between the observed point and all points in the training data, and then selects the k nearest points.
# Majority voting is employed to predict the class of the observed point based on the classes of its nearest neighbors.

# Function to calculate Euclidean distance between observed and actual points
def euclidean_distance(observed:np.ndarray = None, actual:np.ndarray = None):
    return np.linalg.norm(observed - actual, axis=1)

# Function to select top k nearest points based on their distances
def major_k_selection(k: int = None, all_distances: np.ndarray = None):
    return np.argsort(all_distances)[:k]

# Function to find the k nearest points in the training data to the observed inputs
def find_nearest_points(observed_input: list, training_data: np.ndarray, k: int):
    distances = euclidean_distance(observed=np.array([observed_input]), actual=training_data)
    nearest_points_indices = [*major_k_selection(k=k, all_distances=distances)]
    # nearest_points_indices.append(nearest_indices)
    return np.array(nearest_points_indices)

# %%
# Data Preprocessing
# The dataset is shuffled to ensure randomness in the training and testing data.
# 80% of the data is used for training and the remaining 20% for testing.

diabetes_df_shuffled = diabetes_df.sample(frac=1, random_state=42)  # Shuffle with fixed random state for reproducibility

train_proportion = 0.8  # 80% for training, 20% for testing

num_train = int(train_proportion * len(diabetes_df_shuffled))
num_test = len(diabetes_df_shuffled) - num_train

train = diabetes_df_shuffled[:num_train]  # Training set
test = diabetes_df_shuffled[num_train:]  # Testing set

# %%
# Model Implementation and Evaluation
# The kNN algorithm is applied to predict diabetes outcomes for the test data.
# Accuracy is calculated to evaluate the performance of the model.

# Finding distances of all training points from the first observed point in the test set

k = 3  # Set the value of k (number of nearest neighbors)
if input("\n\nDo you want to update 'k' ? [Default k = 3] (y/n) : ").strip().lower() == 'y':
    k = int(input("\tEnter a new value of k : "))
print(f"\nValues of k = {k}\n")

observed = test.iloc[:1, :-1].values.flatten()
print('Default test case:\n', test.iloc[:1, :-1])
columns = test.iloc[:1, :-1].columns.to_list()
if input(f"\n\nDo you want to manually enter a test case (you'll need following: {columns} ) (y/n) : ").strip().lower() == 'y':
    observed = [0.0]*len(columns)
    for index, col in enumerate(columns):
        observed[index] = float(input(f"Enter a floating point value for {col}: "))
    observed = np.array(observed)
print(f"\ntest case : {observed}")

training_data = train.iloc[:, :-1].values
print('Training values:\n', training_data)

nearest_indices = find_nearest_points(observed_input=observed, training_data=training_data, k=k)
print('\nIndices of the', k, 'nearest points in the training data:', nearest_indices)

# Extracting the target variable ('Outcome') from the training data
training_labels = train.iloc[:, -1].values

nearest_labels = training_labels[nearest_indices]
print("\nNearest labels:\n", nearest_labels)

# Predicting the outcome based on majority voting
outcome_counts = np.bincount(nearest_labels.flatten())
predicted_outcome = np.argmax(outcome_counts)
print('\nPredicted outcome based on majority voting:', predicted_outcome)




















# Testing the model
observed_points = test.iloc[:, :-1].values
actual_outcomes = test.iloc[:, -1].values

predicted_outcomes = []
for observed_point in observed_points:
    nearest_indices = find_nearest_points(observed_input=observed_point, training_data=training_data, k=k)
    nearest_labels = training_labels[nearest_indices].flatten()
    outcome_counts = np.bincount(nearest_labels)
    predicted_outcome = np.argmax(outcome_counts)
    predicted_outcomes.append(predicted_outcome)

predicted_outcomes = np.array(predicted_outcomes)
accuracy = np.mean(predicted_outcomes == actual_outcomes) * 100
print('\nAccuracy of the model:', accuracy, '%')

# Diabetes Prediction using k-Nearest Neighbors (kNN)

This project implements the k-Nearest Neighbors (kNN) algorithm for predicting diabetes outcomes based on various patient attributes. The dataset used contains information about diabetes patients, including attributes such as pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

## Project Overview

- The dataset is loaded and explored through statistical visualization using pairplots to understand the relationships between different variables.
- The kNN algorithm is implemented for classification, where the distance between an observed point and all points in the training data is calculated. The algorithm then selects the k nearest points and predicts the class of the observed point based on majority voting.
- Data preprocessing involves shuffling the dataset and splitting it into training and testing sets.
- The model is evaluated by predicting diabetes outcomes for the test data and calculating the accuracy of the predictions.

## Running the Project

### Prerequisites
- Python 3.x
- pip

### Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.

### Environment Setup



# Create a virtual environment (optional)
python -m venv env

# Activate the virtual environment
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate

# Install required libraries
pip install -r requirements.txt


### Running the Code

# Run the main script
python diabetes_prediction.py


## Authors
- [Your Name]

## License
This project is licensed under the [MIT License](LICENSE).

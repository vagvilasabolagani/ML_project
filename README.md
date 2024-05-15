
# Disease Prediction Using Machine Learning

## Project Overview

This Machine Learning project aims to predict diseases based on the symptoms provided by the user. The project leverages three different machine learning algorithms to ensure accurate predictions. The graphical user interface (GUI) for this project is built using Tkinter.

## Table of Contents

- [Project Description](#project-description)
- [Data Sources](#data-sources)
- [Tools and Technologies](#tools-and-technologies)
- [Steps and Procedures](#steps-and-procedures)
  - [Data Collection](#data-collection)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [GUI Development](#gui-development)
- [Usage Instructions](#usage-instructions)
- [Contributors](#contributors)

## Project Description

The objective of this project is to develop a machine learning model that can predict diseases based on user-reported symptoms. By employing multiple machine learning algorithms, the project aims to improve prediction accuracy. A user-friendly GUI is developed using Tkinter to allow users to input symptoms and receive disease predictions.

## Data Sources

- Symptom-disease dataset (source not specified, assume hypothetical or publicly available dataset).
- Data includes a list of symptoms and corresponding diseases.

## Tools and Technologies

- **Python**: Programming language used for the entire project.
- **Scikit-learn**: Library used for implementing machine learning algorithms.
- **Tkinter**: Library used for developing the GUI.
- **Pandas**: Library used for data manipulation and analysis.

## Steps and Procedures

### Data Collection

1. **Gather Data**: Collect a dataset that includes symptoms and corresponding diseases.

### Data Preprocessing

1. **Clean Data**: Handle missing values and normalize the data.
2. **Feature Selection**: Select relevant features (symptoms) for model training.

### Model Training

1. **Algorithm Selection**: Choose three machine learning algorithms for training.
2. **Train Models**: Train the models using the preprocessed dataset.
   - Example algorithms: Decision Tree, Random Forest, and Support Vector Machine (SVM).

### Model Evaluation

1. **Evaluate Models**: Assess the performance of each model using appropriate metrics (e.g., accuracy, precision, recall).
2. **Select Best Model**: Choose the model with the highest performance for deployment.

### GUI Development

1. **Design GUI**: Create a user-friendly interface using Tkinter.
2. **Integrate Models**: Integrate the trained models with the GUI to allow users to input symptoms and receive predictions.

## Usage Instructions

1. **Install Dependencies**: Ensure you have Python and the required libraries installed.
   ```bash
   pip install pandas scikit-learn tkinter
   ```
2. **Run the Application**: Execute the main script to launch the GUI.
   ```bash
   python main.py
   ```
3. **Input Symptoms**: Enter the symptoms in the GUI.
4. **Get Prediction**: Click the predict button to see the predicted disease.


# Predicting Alzheimer's Disease using CNN

This project aims to develop a machine learning model to predict Alzheimer's disease stages using features extracted from MRI imaging and demographic data. The dataset used for this project is the [Alzheimer's Disease 5-Class Dataset from ADNI](https://www.kaggle.com/datasets/madhucharan/alzheimersdisease5classdatasetadni/data).

## Table of Contents
- [Introduction](#introduction)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Model Development](#model-development)
- [Model Evaluation and Selection](#model-evaluation-and-selection)
- [Group Reflections](#group-reflections)
- [Appendix](#appendix)
- [Usage](#usage)

## Introduction

Our objective in this project was to develop a machine learning model to predict Alzheimer's disease stages. The dataset includes cognitive scores, MRI imaging measures, and demographic data divided into five phases of Alzheimer's disease.

## Data Exploration and Preprocessing

We started with an exploratory data analysis (EDA) to understand the distribution and relationships of features. Key steps included:
- Visualizing data distributions and identifying outliers using scatter plots and picture sampling.
- Plotting a correlation matrix to explore connections between features.
- Preprocessing images by resizing to 128x128 pixels, converting to grayscale, and normalizing pixel values between 0 and 1.
- Splitting the data into training (80%) and testing (20%) sets.

For feature engineering, we implemented a `process_images` function to efficiently preprocess images, ensuring robust and error-free data preparation.

## Model Development

We developed a convolutional neural network (CNN) model using TensorFlow and Keras, designed with:
- **Convolutional Layers**: Three layers with ReLU activation to learn spatial hierarchies of features.
- **Pooling Layers**: MaxPooling layers to reduce spatial dimensions and computational load.
- **Dense Layers**: Two dense layers, with ReLU activation and dropout regularization, and a softmax activation for multi-class classification.

## Model Evaluation and Selection

Our model was evaluated using 5-fold cross-validation to ensure generalizability and robustness. Key steps included:
- Implementing model checkpointing to save training states.
- Using Keras Tuner to optimize hyperparameters, employing `RandomSearch` to maximize validation accuracy.
- Retraining the model with optimized hyperparameters to ensure the best performance.

## Group Reflections

Effective division of labour and regular communication were crucial to our success. We tackled challenges in data preprocessing and model optimization, leading to the implementation of a robust CNN model. Key takeaways include the importance of detailed planning, proactive problem-solving, and good organization among team members.

## Appendix

### Dataset
The dataset used in this project can be found [here](https://www.kaggle.com/datasets/madhucharan/alzheimersdisease5classdatasetadni/data).

## Usage

To use the code provided in this repository:
1. Clone the repository to your local machine.
    ```bash
    git clone https://github.com/Fruitkeeper/CNN-to-predict-Alzheimer.git
    ```
2. Navigate to the project directory.
    ```bash
    cd CNN-to-predict-Alzheimer
    ```
3. Run the Jupyter Notebook to see the complete implementation.
    ```bash
    jupyter notebook Alzheimer_project.ipynb
    ```

Feel free to contribute to this project by creating issues or submitting pull requests.

---

**Contributors**:
- Luis Gomez-Acebo
- Leonardo Camilleri
- Noah Valderrama
- Jose Miguel Serrano
- Peter Norbert Karacs

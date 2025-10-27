# Human Activity Recognition (HAR) Model Comparison

This project uses the UCI-HAR (Human Activity Recognition) dataset to classify human activities. It compares the performance of several machine learning and deep learning models, using both the raw time series data and the pre-computed feature set.

## 📋 Overview

The goal is to predict one of six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) performed by a user.

This notebook explores two primary approaches:
1.  **Time Series Model:** A custom multi-input neural network built with Keras that uses the raw inertial sensor data (9 channels, 128 time steps).
2.  **Feature-Based Models:** Several models (a standard ANN, Logistic Regression, SVM, and Random Forest) trained on the 561 pre-engineered features provided by the dataset.

## 📊 Dataset

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
* **Data:** The notebook automatically downloads the `UCI-HAR.zip` file.
* **Features Used:**
    * **Inertial Signals:** The raw time series data from the `Inertial Signals` folder. This consists of 9 files (body/total acceleration and angular velocity for x/y/z axes), each with 128 time steps.
    * **Feature Vector:** The pre-computed `X_train.txt` and `X_test.txt` files, which contain 561 engineered features.

## 🤖 Models and Results

The following models were trained and evaluated on the test set.

| Model | Input Data | Test Accuracy |
| :--- | :--- | :--- |
| **Custom Multi-Input NN** | Raw Time Series (9 channels) | 90.70% |
| **ANN (MLP)** | 561 Features | 95.11% |
| **Logistic Regression** | 561 Features | 94.44% |
| **SVM (RBF Kernel)** | 561 Features | 95.22% |
| **SVM (Linear Kernel)** | 561 Features | **96.10%** |
| **Random Forest** | 561 Features | 93.21% |

### Key Findings

* **Feature Engineering is Key:** All models trained on the 561 pre-engineered features significantly outperformed the custom neural network trained on the raw time series data.
* **Simpler is Better:** The classic machine learning model, **Linear SVM**, achieved the highest accuracy (96.10%) on the test set, slightly outperforming the ANN (95.11%).
* **Overfitting:** The Random Forest classifier achieved 100% accuracy on the training data but only 93.21% on the test data, indicating significant overfitting.

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/UCI-HAR-Classification-Comparison.git](https://github.com/YOUR_USERNAME/UCI-HAR-Classification-Comparison.git)
    cd UCI-HAR-Classification-Comparison
    ```
2.  **Install dependencies:**
    This project requires `tensorflow`, `pandas`, `scikit-learn`, `numpy`, and `matplotlib`.
    ```bash
    pip install tensorflow pandas scikit-learn numpy matplotlib
    ```
3.  **Run the Notebook:**
    Open and run the `time_series.ipynb` notebook in a Jupyter or Google Colab environment. The notebook will automatically download the required dataset files.

## Libraries Used

* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

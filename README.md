# 🚚 Porter Delivery Time Prediction

A **machine learning-powered delivery time prediction application** built with **Python, TensorFlow/Keras, Scikit-learn, and Streamlit**.

The application uses historical Porter order data to predict the estimated delivery time for a new order based on store, order, pricing, and operational factors.

---

## 📌 Project Overview

Delivery time is an important factor in logistics and food-delivery operations. This project uses historical Porter order data to build a regression model capable of estimating how long an order will take to be delivered.

The project combines:

* Data preprocessing
* Feature engineering
* Missing-value handling
* Categorical feature encoding
* Feature scaling
* Neural-network regression
* Model evaluation
* Interactive Streamlit deployment

The final application provides a simple interface where users can enter order-related information and receive an estimated delivery time in minutes.

---

## ✨ Features

* 🚚 Delivery time prediction
* 🧠 TensorFlow/Keras neural-network regression model
* 📊 Historical Porter order data
* 🔄 Data preprocessing and feature engineering
* 🧹 Missing-value handling
* 🔤 Categorical feature encoding
* 📏 Feature scaling using `StandardScaler`
* 📈 Model evaluation using regression metrics
* 🖥️ Interactive Streamlit interface
* ⚡ Real-time prediction

---

## 🏗️ Project Workflow

```text
Historical Porter Dataset
          │
          ▼
   Data Preprocessing
          │
          ▼
   Feature Engineering
          │
          ▼
 Missing Value Handling
          │
          ▼
 Categorical Encoding
          │
          ▼
    Feature Scaling
          │
          ▼
 Train / Validation / Test Split
          │
          ▼
 TensorFlow / Keras Model
          │
          ▼
    Model Evaluation
          │
          ▼
     Streamlit App
          │
          ▼
 Delivery Time Prediction
```

---

## 📊 Dataset

The project uses historical Porter order data stored in:

```text
porter.csv
```

The dataset contains order and operational information such as:

* Store category
* Order protocol
* Number of items
* Subtotal
* Minimum item price
* Maximum item price
* Number of on-shift delivery partners
* Number of busy delivery partners
* Number of outstanding orders
* Order creation time
* Actual delivery time

The target variable is derived from the difference between:

```text
actual_delivery_time - created_at
```

This represents the delivery duration.

---

## 🔑 Important Features

Some of the major features used by the model include:

| Feature                    | Description                          |
| -------------------------- | ------------------------------------ |
| `store_primary_category`   | Primary category of the store        |
| `order_protocol`           | Order placement protocol             |
| `total_items`              | Total number of items                |
| `subtotal`                 | Order subtotal                       |
| `num_distinct_items`       | Number of distinct items             |
| `min_item_price`           | Minimum item price                   |
| `max_item_price`           | Maximum item price                   |
| `total_onshift_partners`   | Delivery partners currently on shift |
| `total_busy_partners`      | Busy delivery partners               |
| `total_outstanding_orders` | Outstanding orders                   |

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes several steps.

### 1. Missing Value Handling

Missing numerical values are handled using appropriate statistical values, while categorical values are filled using available category information.

### 2. Categorical Encoding

Categorical variables such as:

```text
store_primary_category
order_protocol
```

are converted into numerical representations so that they can be processed by the machine-learning model.

### 3. Feature Scaling

Numerical features are scaled using:

```python
StandardScaler()
```

The scaler is fitted on the training data and then applied to validation and test data.

### 4. Target Creation

Delivery time is calculated from the order timestamps:

```text
Delivery Time = Actual Delivery Time - Created Time
```

The resulting value is represented in minutes.

---

## 🧠 Machine Learning Model

The project uses a **TensorFlow/Keras neural network** for regression.

The model learns the relationship between order characteristics and delivery duration.

Conceptually:

```text
Order Features
      │
      ▼
Dense Neural Network
      │
      ▼
Learned Patterns
      │
      ▼
Predicted Delivery Time
```

The final output is a continuous numerical value representing the estimated delivery time in minutes.

---

## 📐 Model Evaluation

The model can be evaluated using standard regression metrics such as:

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted delivery time.

```text
MAE = Average |Actual - Predicted|
```

### Root Mean Squared Error — RMSE

Penalizes larger prediction errors more strongly.

### R² Score

Measures how much of the variation in delivery time is explained by the model.

---

## 🖥️ Streamlit Application

The trained model is integrated into a Streamlit application.

The application allows users to:

1. Provide order-related information
2. Process the input
3. Apply the same preprocessing used during training
4. Generate a prediction
5. Display the estimated delivery duration

Example:

```text
Input Order
     │
     ▼
Preprocessing
     │
     ▼
Trained Model
     │
     ▼
Prediction
     │
     ▼
Estimated Delivery Time
```

---

## 📁 Project Structure

```text
Porter-Project/
│
├── app.py
├── porter.csv
├── PorterCase.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* TensorFlow
* Keras
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Encoding

* category_encoders

### Deployment / UI

* Streamlit

### Development

* Jupyter Notebook
* VS Code

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project:

```bash
cd Porter-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

---

## 📓 Jupyter Notebook

The complete machine-learning workflow and experimentation are available in:

```text
PorterCase.ipynb
```

The notebook contains the data analysis, preprocessing, model development, training, and evaluation workflow used to develop the solution.

---

## 🔮 Example Prediction

The application takes order-related information such as:

```text
Store Category
Order Protocol
Total Items
Subtotal
Item Prices
Delivery Partner Availability
Outstanding Orders
```

and produces an estimated delivery duration:

```text
Estimated Delivery Time: XX minutes
```

---

## 🎯 Project Objective

The primary objective of this project is to demonstrate how machine learning can be applied to **delivery-time estimation and operational decision-making**.

The project brings together the complete machine-learning workflow:

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Evaluation
 ↓
Deployment
 ↓
Prediction
```

---

## 🚀 Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Advanced feature engineering
* Model comparison with XGBoost, Random Forest, and other regression models
* Cross-validation
* Improved outlier treatment
* Model explainability using SHAP
* Model serialization and versioning
* Cloud deployment
* Prediction monitoring
* Automated retraining pipeline

---

## 👨‍💻 Author

**Safi Siddiqui**

MERN Stack Developer | Aspiring Data Scientist | AI Engineer

---

## ⭐ Acknowledgement

This project was developed as a machine-learning case study focused on predicting delivery duration from historical order and operational data.

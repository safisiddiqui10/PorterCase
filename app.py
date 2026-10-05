import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import os

# Page configuration
st.set_page_config(page_title="Porter Delivery Time Predictor", layout="centered")

# -----------------------------------------------------------------------------
# Resource Loading and Preprocessing Setup
# -----------------------------------------------------------------------------

@st.cache_resource
def load_resources():
    """
    Loads the model, dataset, and fits the necessary transformers.
    This is cached so it only happens once per session.
    """
    model_path = "tf_model.h5"
    data_path = "porter.csv"

    # Load Model
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found!")
        model = None
    else:
        model = load_model(model_path)

    # Load Data for Fitting Transformers
    if not os.path.exists(data_path):
        st.error(f"Dataset file {data_path} not found!")
        df = None
    else:
        df = pd.read_csv(data_path)

    if df is None:
        return model, None, None, None, None, None

    # 1. Calculate target: delivery_time
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'], errors='coerce')
    df['delivery_time'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60
    df['delivery_time'] = df['delivery_time'].fillna(df['delivery_time'].mean())

    # 2. Imputation values
    # Categorical modes
    store_primary_cat_mode = df['store_primary_category'].mode()[0]
    order_protocol_mode = df['order_protocol'].mode()[0]

    # Numerical medians
    num_cols_impute = ['total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']
    medians = {col: df[col].median() for col in num_cols_impute}

    # 3. Fit TargetEncoder
    encoder = TargetEncoder()
    cols_to_encode = ['store_id', 'store_primary_category']
    encoder.fit(df[cols_to_encode], df['delivery_time'])

    # 4. Fit StandardScaler
    # The model expects features in this exact order (matching notebook's X):
    # ['store_primary_category', 'order_protocol', 'total_items', 'subtotal',
    #  'num_distinct_items', 'min_item_price', 'max_item_price',
    #  'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders']

    df_encoded = df.copy()
    df_encoded[cols_to_encode] = encoder.transform(df[cols_to_encode])

    feature_cols = [
        'store_primary_category', 'order_protocol', 'total_items', 'subtotal',
        'num_distinct_items', 'min_item_price', 'max_item_price',
        'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders'
    ]

    # Fill NaNs in the feature set for scaling
    df_encoded['store_primary_category'] = df_encoded['store_primary_category'].fillna(df_encoded['store_primary_category'].median())
    df_encoded['order_protocol'] = df_encoded['order_protocol'].fillna(order_protocol_mode)
    for col in num_cols_impute:
        df_encoded[col] = df_encoded[col].fillna(medians[col])

    scaler = StandardScaler()
    scaler.fit(df_encoded[feature_cols])

    return model, encoder, scaler, medians, store_primary_cat_mode, order_protocol_mode

model, encoder, scaler, medians, cat_mode, protocol_mode = load_resources()

# -----------------------------------------------------------------------------
# User Interface
# -----------------------------------------------------------------------------

st.title("🚚 Porter Delivery Time Predictor")
st.markdown("Enter order details to predict the estimated delivery time.")

# We need unique categories for the dropdown
@st.cache_data
def get_categories():
    df = pd.read_csv("porter.csv")
    return sorted(df['store_primary_category'].dropna().unique().tolist())

try:
    categories = get_categories()
except Exception:
    categories = []

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        store_id = st.text_input("Store ID", value="f0ade77b43923b38237db569b016ba25")
        store_cat = st.selectbox("Store Primary Category", options=categories)
        order_protocol = st.number_input("Order Protocol", value=1.0)
        total_items = st.number_input("Total Items", value=1, step=1)
        subtotal = st.number_input("Subtotal", value=1000.0)
        num_distinct = st.number_input("Number of Distinct Items", value=1, step=1)

    with col2:
        min_price = st.number_input("Min Item Price", value=100.0)
        max_price = st.number_input("Max Item Price", value=500.0)
        onshift = st.number_input("Total On-shift Partners", value=20.0)
        busy = st.number_input("Total Busy Partners", value=10.0)
        outstanding = st.number_input("Total Outstanding Orders", value=15.0)

    submit = st.form_submit_button("Predict Delivery Time")

# -----------------------------------------------------------------------------
# Prediction Pipeline
# -----------------------------------------------------------------------------

if submit:
    if model is None:
        st.error("Model not loaded. Prediction unavailable.")
    else:
        # 1. Create DataFrame from inputs
        input_data = pd.DataFrame([{
            'store_id': store_id,
            'store_primary_category': store_cat,
            'order_protocol': order_protocol,
            'total_items': total_items,
            'subtotal': subtotal,
            'num_distinct_items': num_distinct,
            'min_item_price': min_price,
            'max_item_price': max_price,
            'total_onshift_partners': onshift,
            'total_busy_partners': busy,
            'total_outstanding_orders': outstanding
        }])

        # 2. Imputation (for safety, though Streamlit inputs have defaults)
        input_data['store_primary_category'] = input_data['store_primary_category'].fillna(cat_mode)
        input_data['order_protocol'] = input_data['order_protocol'].fillna(protocol_mode)
        for col, val in medians.items():
            input_data[col] = input_data[col].fillna(val)

        # 3. Target Encoding
        cols_to_encode = ['store_id', 'store_primary_category']
        input_data[cols_to_encode] = encoder.transform(input_data[cols_to_encode])

        # 4. Scaling
        feature_cols = [
            'store_primary_category', 'order_protocol', 'total_items', 'subtotal',
            'num_distinct_items', 'min_item_price', 'max_item_price',
            'total_onshift_partners', 'total_busy_partners', 'total_outstanding_orders'
        ]
        X_scaled = scaler.transform(input_data[feature_cols])

        # 5. Predict
        prediction = model.predict(X_scaled, verbose=0)
        delivery_time = float(prediction[0][0])

        st.success(f"### Predicted Delivery Time: {delivery_time:.2f} minutes")

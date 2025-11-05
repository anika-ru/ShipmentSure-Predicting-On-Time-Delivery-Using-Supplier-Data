import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model and Preprocessor ---
# NOTE: Ensure these two files are in the same directory as this script!
try:
    preprocessor = joblib.load('preprocessor_pipeline.joblib')
    model = joblib.load('best_model_LogisticRegression.joblib')
except FileNotFoundError:
    st.error("Deployment files not found. Please ensure 'preprocessor_pipeline.joblib' and 'best_model_LogisticRegression.joblib' are in the same directory.")
    st.stop()


# --- 2. Define Prediction Function ---
def predict_shipment(data):
    """
    Takes a DataFrame of raw input data and predicts 
    Reached.on.Time_Y.N using the saved pipeline, applying a custom threshold.
    """
    try:
        # 1. Transform the raw data using the saved ColumnTransformer
        transformed_data = preprocessor.transform(data)
        
        # 2. Get probability for the positive class (class 1: Reached on Time)
        probability = model.predict_proba(transformed_data)[0][1]
        
        # 3. Apply custom decision threshold (0.45)
        # This makes the model more sensitive to predicting delays (Class 0)
        DECISION_THRESHOLD = 0.45 
        
        if probability >= DECISION_THRESHOLD:
            prediction = 1 # Reached on Time
        else:
            prediction = 0 # Delayed

        return prediction, probability
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None


# --- 3. Streamlit UI Design ---
st.set_page_config(page_title="Shipment Sure Predictor", layout="wide")

st.title("📦 Shipment Sure: On-Time Delivery Predictor")
st.markdown("### Powered by Logistic Regression (ROC AUC: 0.9611)")
st.markdown("🚚 **ShipmentSure helps ensure accurate, efficient, and secure tracking and management of shipments throughout the delivery process.**")
st.write("---")

# --- Tabs for Predictor and Model Info ---
tab1, tab2 = st.tabs(["🚀 Predictor", "🧠 Model Info"])

# =========================================================================
# === TAB 1: PREDICTOR INTERFACE AND LOGIC ================================
# =========================================================================
with tab1:
    col1, col2 = st.columns(2)

    # --- Feature Input Fields (LEFT COLUMN) ---
    with col1:
        st.header("Shipment Details")
        
        warehouse_block = st.selectbox(
            "Warehouse Block (Location A-F):",
            options=['A', 'B', 'C', 'D', 'E', 'F']
        )
        
        mode_of_shipment = st.selectbox(
            "Mode of Shipment:",
            options=['Flight', 'Road', 'Ship']
        )
        
        product_importance = st.selectbox(
            "Product Importance:",
            options=['low', 'medium', 'high']
        )
        
        gender = st.selectbox(
            "Customer Gender:",
            options=['F', 'M']
        )

    # --- Feature Input Fields (RIGHT COLUMN) ---
    with col2:
        st.header("Customer & Product Metrics")
        
        cost_of_the_product = st.number_input(
            "Cost of the Product ($):", 
            min_value=10, max_value=500, value=200, step=1
        )
        
        weight_in_gms = st.number_input(
            "Weight (grams):", 
            min_value=100, max_value=8000, value=4000, step=100
        )
        
        customer_care_calls = st.slider(
            "Customer Care Calls:", 
            min_value=1, max_value=7, value=3
        )
        
        customer_rating = st.slider(
            "Customer Rating (1-5):", 
            min_value=1, max_value=5, value=3
        )
        
        prior_purchases = st.slider(
            "Prior Purchases:", 
            min_value=1, max_value=10, value=3
        )
        
        discount_offered = st.slider(
            "Discount Offered (%):", 
            min_value=0, max_value=65, value=10
        )


    # --- Prediction Button and Logic ---
    if st.button("Predict On-Time Delivery"):
        # 1. Gather all inputs into a DataFrame
        input_data = pd.DataFrame({
            'Warehouse_block': [warehouse_block], 
            'Mode_of_Shipment': [mode_of_shipment], 
            'Customer_care_calls': [customer_care_calls], 
            'Customer_rating': [customer_rating], 
            'Cost_of_the_Product': [cost_of_the_product], 
            'Prior_purchases': [prior_purchases], 
            'Product_importance': [product_importance], 
            'Gender': [gender], 
            'Discount_offered': [discount_offered], 
            'Weight_in_gms': [weight_in_gms]
        })
        
        # 2. Make prediction
        prediction, probability = predict_shipment(input_data)
        
        st.write("---")
        st.header("Prediction Result:")
        
        if prediction is not None:
            if prediction == 1:
                st.success(f"**Prediction: SHIPMENT WILL REACH ON TIME!**")
                result_text = "The model predicts the shipment will be delivered successfully on time. This is a low-risk shipment."
            else:
                st.warning(f"**Prediction: SHIPMENT WILL BE DELAYED.**")
                result_text = "The model suggests this shipment is likely to be delayed. **Immediate action is recommended.**"

            # Display probability
            st.info(f"Confidence Score (Probability of On-Time): **{probability * 100:.2f}%**")
            
            st.write(result_text)


# =========================================================================
# === TAB 2: MODEL DESCRIPTION ============================================
# =========================================================================
with tab2:
    st.markdown("""
        ### 🧠 Model Architecture

        This prediction engine is powered by a **Logistic Regression** classifier. It was selected after a rigorous, comprehensive comparative analysis against seven other classification models (including XGBoost, SVC, and Random Forest).

        The final choice was driven by the model's optimal balance between high discriminatory performance and exceptional computational speed, making it the most practical choice for real-time logistics deployment.

        ---

        #### Key Performance Metrics (on Unseen Test Data):

        | Metric | Score | Interpretation |
        | :--- | :--- | :--- |
        | **Test ROC AUC** | **0.9611** | High ability to distinguish between on-time and delayed shipments across all thresholds. |
        | **Test Accuracy** | 0.9523 | 95.23% overall correct predictions. |
        | **F1-Score (Class 1)** | 0.9511 | Strong balance between Precision and Recall for the "On Time" prediction. |
        | **Processing Time** | **0.07 s** | Extremely fast and efficient, ensuring low latency in predictions. |

        ---

        ### How to Interpret the Prediction:

        The model uses processed input features (Weight, Discount Offered, Customer Calls, etc.) to calculate the probability of an on-time delivery.

        * **Prediction:** The final YES/NO classification is determined by a **custom decision threshold (set at 45%)**. This adjustment forces the model to be more cautious and sensitive to potential delays (Class 0), improving its utility for proactive risk management.
        * **Confidence Score:** This is the direct probability (e.g., 85% confidence of being on time). A low score (e.g., 13.17%) immediately flags a high-risk shipment, requiring immediate intervention.
    """)

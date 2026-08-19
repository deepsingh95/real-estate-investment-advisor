import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

# ================= LOAD MODELS =================
@st.cache_resource
def load_artifacts():
    classification_model = joblib.load('models/classification_model.pkl')
    regression_model = joblib.load('models/regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    scaler2 = joblib.load('models/scaler2.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_cols_class = joblib.load('models/feature_cols_class.pkl')
    feature_cols_reg = joblib.load('models/feature_cols_reg.pkl')
    return classification_model, regression_model, scaler, scaler2, label_encoders, feature_cols_class, feature_cols_reg

classification_model, regression_model, scaler, scaler2, label_encoders, feature_cols_class, feature_cols_reg = load_artifacts()

@st.cache_data
def load_data():
    return pd.read_csv('data/india_housing_prices.csv')

df = load_data()

# ================= TITLE =================
st.title("🏠 Real Estate Investment Advisor")
st.markdown("Predict property investment potential and future value using Machine Learning")

# ================= SIDEBAR: USER INPUT FORM =================
st.sidebar.header("Enter Property Details")

state = st.sidebar.selectbox("State", sorted(df['State'].unique()))
city = st.sidebar.selectbox("City", sorted(df[df['State']==state]['City'].unique()))
property_type = st.sidebar.selectbox("Property Type", sorted(df['Property_Type'].unique()))
bhk = st.sidebar.slider("BHK", 1, 5, 2)
size_sqft = st.sidebar.number_input("Size (SqFt)", min_value=500, max_value=5000, value=1500)
price_lakhs = st.sidebar.number_input("Current Price (Lakhs)", min_value=10.0, max_value=500.0, value=100.0)
year_built = st.sidebar.slider("Year Built", 1990, 2023, 2010)
floor_no = st.sidebar.slider("Floor Number", 0, 30, 5)
total_floors = st.sidebar.slider("Total Floors", 1, 30, 10)
furnished_status = st.sidebar.selectbox("Furnished Status", sorted(df['Furnished_Status'].unique()))
nearby_schools = st.sidebar.slider("Nearby Schools", 1, 10, 5)
nearby_hospitals = st.sidebar.slider("Nearby Hospitals", 1, 10, 5)
transport = st.sidebar.selectbox("Public Transport Accessibility", ['Low', 'Medium', 'High'])
parking = st.sidebar.selectbox("Parking Space", ['Yes', 'No'])
security = st.sidebar.selectbox("Security", ['Yes', 'No'])
facing = st.sidebar.selectbox("Facing", sorted(df['Facing'].unique()))
owner_type = st.sidebar.selectbox("Owner Type", sorted(df['Owner_Type'].unique()))
availability = st.sidebar.selectbox("Availability Status", sorted(df['Availability_Status'].unique()))
amenities_count = st.sidebar.slider("Number of Amenities", 1, 6, 3)

predict_btn = st.sidebar.button("🔍 Predict")

# ================= PREPARE INPUT =================
def prepare_input():
    age = 2026 - year_built
    # IMPORTANT: training data's Price_per_SqFt is in Lakhs-per-SqFt (NOT rupees),
    # so we must NOT multiply by 100000 here — must match training units exactly.
    price_per_sqft = price_lakhs / size_sqft
    transport_map = {'Low': 0, 'Medium': 1, 'High': 2}

    input_dict = {
        'BHK': bhk,
        'Size_in_SqFt': size_sqft,
        'Price_per_SqFt': price_per_sqft,
        'Year_Built': year_built,
        'Floor_No': floor_no,
        'Total_Floors': total_floors,
        'Age_of_Property': age,
        'Nearby_Schools': nearby_schools,
        'Nearby_Hospitals': nearby_hospitals,
        'Security_encoded': 1 if security == 'Yes' else 0,
        'Parking_encoded': 1 if parking == 'Yes' else 0,
        'Transport_encoded': transport_map[transport],
        'Amenities_Count': amenities_count,
        'State_encoded': label_encoders['State'].transform([state])[0],
        'City_encoded': label_encoders['City'].transform([city])[0],
        'Property_Type_encoded': label_encoders['Property_Type'].transform([property_type])[0],
        'Furnished_Status_encoded': label_encoders['Furnished_Status'].transform([furnished_status])[0],
        'Facing_encoded': label_encoders['Facing'].transform([facing])[0],
        'Owner_Type_encoded': label_encoders['Owner_Type'].transform([owner_type])[0],
    }
    return input_dict, price_per_sqft

# ================= MAIN PANEL =================
if predict_btn:
    input_dict, price_per_sqft = prepare_input()

    # ---- Classification ----
    X_class = pd.DataFrame([input_dict])[feature_cols_class]
    good_investment = classification_model.predict(X_class)[0]
    confidence = classification_model.predict_proba(X_class)[0][good_investment]

    # ---- Regression ----
    input_dict_reg = dict(input_dict)
    input_dict_reg['Price_in_Lakhs'] = price_lakhs
    X_reg = pd.DataFrame([input_dict_reg])[feature_cols_reg]
    X_reg_scaled = scaler2.transform(X_reg)
    future_price = regression_model.predict(X_reg_scaled)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Investment Classification")
        if good_investment == 1:
            st.success(f"✅ GOOD INVESTMENT")
        else:
            st.error(f"❌ NOT A GOOD INVESTMENT")
        st.metric("Confidence Score", f"{confidence*100:.1f}%")

    with col2:
        st.subheader("💰 Price Forecast (5 Years)")
        st.metric("Current Price", f"₹{price_lakhs:.2f} Lakhs")
        st.metric("Estimated Price (5Y)", f"₹{future_price:.2f} Lakhs", 
                   delta=f"+{future_price - price_lakhs:.2f} Lakhs")

    st.divider()

    # ---- Feature Importance ----
    st.subheader("🔑 Feature Importance (Classification Model)")
    importance_df = pd.DataFrame({
        'Feature': feature_cols_class,
        'Importance': classification_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title("Top 10 Important Features")
    st.pyplot(fig)

else:
    st.info("👈 Fill in the property details in the sidebar and click 'Predict' to get insights.")

# ================= VISUAL INSIGHTS SECTION =================
st.divider()
st.header("📈 Market Insights")

tab1, tab2, tab3 = st.tabs(["City-wise Price Heatmap", "Price Distribution", "BHK Analysis"])

with tab1:
    city_avg = df.groupby('City')['Price_per_SqFt'].mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=city_avg.values, y=city_avg.index, hue=city_avg.index, palette='viridis', legend=False, ax=ax)
    ax.set_title("Top 15 Cities by Avg Price per SqFt")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True, ax=ax)
    ax.set_title("Overall Price Distribution")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(data=df, x='BHK', hue='BHK', palette='Set2', legend=False, ax=ax)
    ax.set_title("BHK Distribution")
    st.pyplot(fig)
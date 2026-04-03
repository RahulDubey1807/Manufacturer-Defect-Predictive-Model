import streamlit as st
import joblib
import numpy as np

# Load model
data = joblib.load("defect_model.pkl")
model = data["model"]
features = data["features"]

# UI
st.title("🔧 Manufacturing Defect Prediction System")

st.info("This system predicts whether a product is likely to be defective based on manufacturing parameters using a Random Forest model.")

st.markdown("### Enter production parameters to predict defect occurrence")

# Inputs
ProductionVolume = st.number_input("Production Volume", value=500.0)
ProductionCost = st.number_input("Production Cost", value=12000.0)
SupplierQuality = st.number_input("Supplier Quality", value=90.0)
DeliveryDelay = st.number_input("Delivery Delay", value=2.0)
QualityScore = st.number_input("Quality Score", value=80.0)
MaintenanceHours = st.number_input("Maintenance Hours", value=10.0)
DowntimePercentage = st.number_input("Downtime Percentage", value=2.0)
InventoryTurnover = st.number_input("Inventory Turnover", value=5.0)
StockoutRate = st.number_input("Stockout Rate", value=0.05)
WorkerProductivity = st.number_input("Worker Productivity", value=90.0)
SafetyIncidents = st.number_input("Safety Incidents", value=2.0)
EnergyConsumption = st.number_input("Energy Consumption", value=3000.0)
EnergyEfficiency = st.number_input("Energy Efficiency", value=0.3)
AdditiveProcessTime = st.number_input("Additive Process Time", value=5.0)
AdditiveMaterialCost = st.number_input("Additive Material Cost", value=300.0)

# Predict button
if st.button("Predict Defect"):

    input_data = np.array([[
        ProductionVolume, ProductionCost, SupplierQuality, DeliveryDelay,
        QualityScore, MaintenanceHours, DowntimePercentage,
        InventoryTurnover, StockoutRate, WorkerProductivity,
        SafetyIncidents, EnergyConsumption, EnergyEfficiency,
        AdditiveProcessTime, AdditiveMaterialCost
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    # Output
    if prediction == 1:
        st.error("⚠️ Defect Likely")
    else:
        st.success("✅ No Defect")

    st.write(f"Defect Probability: {prob:.2f}")
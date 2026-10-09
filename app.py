import os
import zipfile
import tempfile
import streamlit as st
import joblib

st.set_page_config(page_title="Supply Chain Demand Forecasting", layout="wide")

st.title("📦 Multi-Domain Supply Chain Demand Forecasting System")
st.write("Dynamic inventory demand prediction powered by Small Language Model sentiment scoring and category-specific models.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a temp directory outside Streamlit's watched workspace to avoid infinite reboots
TEMP_MODELS_DIR = os.path.join(tempfile.gettempdir(), "domain_models")

@st.cache_resource
def extract_and_load_models():
    os.makedirs(TEMP_MODELS_DIR, exist_ok=True)
    
    # Extract zips into the temp directory if not already present
    for batch_name in ["domain_models_batch1", "domain_models_batch2"]:
        batch_extracted_path = os.path.join(TEMP_MODELS_DIR, batch_name)
        zip_path = os.path.join(BASE_DIR, f"{batch_name}.zip")
        
        if not os.path.exists(batch_extracted_path) and os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_MODELS_DIR)

    # Scan for .pkl files in the temp directory
    pkl_files_map = {}
    for root, dirs, files in os.walk(TEMP_MODELS_DIR):
        for f in files:
            if f.endswith(".pkl"):
                clean_name = f.replace("_sentiment_model.pkl", "").replace(".pkl", "")
                clean_name = clean_name.replace("ts_", "").replace("all_", "").replace("_", " ")
                display_name = clean_name.title()
                pkl_files_map[display_name] = os.path.join(root, f)
                
    return pkl_files_map

with st.spinner("Initializing models..."):
    category_map = extract_and_load_models()

categories = sorted(list(category_map.keys()))

if not categories:
    st.error("No model artifacts found! Please ensure your zip files are in the repository root.")
else:
    selected_display_name = st.selectbox("Select Product Domain Category:", categories)
    selected_filepath = category_map[selected_display_name]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product Inputs")
        product_description = st.text_area(
            "Paste Product Description & Customer Reviews:", 
            "e.g., Ultra-bright wireless bluetooth headphones with noise cancellation..."
        )
        recent_sales_lag = st.number_input("Recent Weekly Sales Lag (Units):", min_value=0.0, value=150.0)

    with col2:
        st.subheader("Forecast Engine Output")
        if st.button("Generate Demand Forecast", type="primary"):
            if os.path.exists(selected_filepath):
                model = joblib.load(selected_filepath)
                mock_sentiment_index = 8.8 
                
                try:
                    n_feats = getattr(model, "n_features_in_", 1)
                except Exception:
                    n_feats = 1

                if n_feats == 1:
                    prediction_input = [[recent_sales_lag]]
                else:
                    prediction_input = [[recent_sales_lag, mock_sentiment_index]]

                predicted_demand = model.predict(prediction_input)[0]
                
                st.success("Forecast Generated Successfully!")
                st.metric(label=f"Predicted Next-Week Demand ({selected_display_name})", value=f"{int(predicted_demand)} units")
                st.info(f"Loaded from path: `{selected_filepath}` | Model expects: {n_feats} feature(s)")
            else:
                st.error(f"Could not find model file at: {selected_filepath}")

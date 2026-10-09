import os
import streamlit as st
import joblib

st.set_page_config(page_title="Supply Chain Demand Forecasting", layout="wide")

st.title("📦 Multi-Domain Supply Chain Demand Forecasting System")
st.write("Dynamic inventory demand prediction powered by Small Language Model sentiment scoring and category-specific models.")

# --- SET YOUR ABSOLUTE PROJECT FOLDER PATH HERE ---
BASE_DIR = "/Users/naitripanchal/Downloads/multimodal-demand-forecasting-slm"

@st.cache_resource
def load_available_categories():
    pkl_files_map = {}
    if os.path.exists(BASE_DIR):
        for root, dirs, files in os.walk(BASE_DIR):
            for f in files:
                if f.endswith(".pkl"):
                    clean_name = f.replace("_sentiment_model.pkl", "").replace(".pkl", "")
                    clean_name = clean_name.replace("ts_", "").replace("all_", "").replace("_", " ")
                    display_name = clean_name.title()
                    pkl_files_map[display_name] = os.path.join(root, f)
    return pkl_files_map

category_map = load_available_categories()
categories = sorted(list(category_map.keys()))

if not categories:
    st.error(f"No model artifacts found in: `{BASE_DIR}`. Please verify your folder path.")
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
                # Lazy-load the model
                model = joblib.load(selected_filepath)
                mock_sentiment_index = 8.8 
                
                # Check how many features this specific model was trained with
                n_features = getattr(model, "n_features_in_", 1)

                # Dynamically format input to match what the model expects
                if n_features == 1:
                    prediction_input = [[recent_sales_lag]]
                else:
                    prediction_input = [[recent_sales_lag, mock_sentiment_index]]

                predicted_demand = model.predict(prediction_input)[0]
                
                st.success("Forecast Generated Successfully!")
                st.metric(label=f"Predicted Next-Week Demand ({selected_display_name})", value=f"{int(predicted_demand)} units")
                st.info(f"Loaded from path: `{selected_filepath}` | Model expects: {n_features} feature(s)")
            else:
                st.error(f"Could not find model file at: {selected_filepath}")
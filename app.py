import streamlit as st
import pandas as pd
import joblib

st.title("Sandstorm Prediction System")
st.write("Upload the satellite data CSV file")

uploaded_file = st.file_uploader("Upload CSV file")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Satellite Data")
    st.dataframe(df)

    model = joblib.load("sandstorm_model_4features.pkl")

    df_model = df.rename(columns={
        "DHT temperature": "Temp_C",
        "Dust": "Estimated_Dust",
        "Humadity": "Rel Hum_%",
        "Pressure": "Press_kPa"
    })

    X = df_model[["Temp_C", "Estimated_Dust", "Rel Hum_%", "Press_kPa"]]

    # prediction
    predictions = model.predict(X)

    df["Prediction"] = predictions

    df["Result"] = df["Prediction"].map({
        0: "No Sandstorm",
        1: "Sandstorm"
    })

    st.subheader("Prediction Results")
    st.dataframe(df)

    sandstorm_count = (df["Prediction"] == 1).sum()
    no_sandstorm_count = (df["Prediction"] == 0).sum()

    col1, col2 = st.columns(2)

    col1.metric("Sandstorms Detected", sandstorm_count)
    col2.metric("No Sandstorms", no_sandstorm_count)
    
    chart_data = pd.DataFrame({
        "Type": ["Sandstorm", "No Sandstorm"],
        "Count": [sandstorm_count, no_sandstorm_count]
    })

    st.bar_chart(chart_data.set_index("Type"))

    if sandstorm_count > 0:
        st.error("⚠ Sandstorm Detected in Satellite Data")
    else:
        st.success("✅ No Sandstorms Detected")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download prediction results",
        data=csv,
        file_name="sandstorm_predictions.csv",
        mime="text/csv"
    )
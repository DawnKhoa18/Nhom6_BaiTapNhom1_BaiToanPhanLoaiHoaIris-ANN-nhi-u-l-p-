
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# =========================
# Load model và preprocessing
# =========================

model = tf.keras.models.load_model("iris_ann_model.h5")

scaler = joblib.load("scaler.pkl")

label_encoder = joblib.load("label_encoder.pkl")

# =========================
# Giao diện
# =========================

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Phân Loại Hoa Iris Bằng ANN")

st.write("Nhập thông số của hoa để AI dự đoán loại hoa.")

# =========================
# Input
# =========================

sepal_length = st.number_input(
    "Sepal Length",
    min_value=4.0,
    max_value=8.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=2.0,
    max_value=5.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    min_value=1.0,
    max_value=7.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.1,
    max_value=3.0,
    value=0.2
)

# =========================
# Predict Button
# =========================

if st.button("Dự đoán"):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Chuẩn hóa dữ liệu
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    predicted_class = np.argmax(prediction)

    flower_name = label_encoder.inverse_transform([predicted_class])[0]

    confidence = np.max(prediction) * 100

    # Hiển thị kết quả
    st.success(f"🌼 Loại hoa dự đoán: {flower_name}")

    st.info(f"🎯 Độ tin cậy: {confidence:.2f}%")

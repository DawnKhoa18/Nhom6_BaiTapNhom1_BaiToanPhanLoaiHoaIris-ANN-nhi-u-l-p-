import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from PIL import Image

# 1. Cấu hình giao diện Trang Web
st.set_page_config(
    page_title="Phân loại Hoa Iris - Nhóm 6",
    page_icon="🌸",
    layout="wide" # Chuyển sang giao diện rộng (wide) để chia cột cho đẹp
)

# 2. Hàm tải các file Model và Tiền xử lý
@st.cache_resource
def load_prediction_resources():
    model = load_model("ann_iris_model.keras", compile=False) # Bỏ qua compile để tránh lỗi phiên bản
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_prediction_resources()
except Exception as e:
    st.error(f"Lỗi tải file model: {e}")
    st.stop()

# 3. Thiết kế Giao diện UI
st.title("🌸 Ứng Dụng Phân Loại Hoa Iris (ANN Nhiều Lớp)")
st.write("Sản phẩm được phát triển bởi **Nhóm 6**.")
st.markdown("---")

# Chia màn hình làm 2 cột chính: Cột trái nhập số - Cột phải xử lý hình ảnh
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.subheader("📥 Nhập thông số đặc trưng (cm):")
    
    # Tạo 2 cột nhỏ bên trong để nhập số gọn gàng
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Chiều dài đài hoa (Sepal Length)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Chiều rộng đài hoa (Sepal Width)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    with col2:
        petal_length = st.number_input("Chiều dài cánh hoa (Petal Length)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Chiều rộng cánh hoa (Petal Width)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    
    st.markdown("---")
    
    # Nút bấm dự đoán
    if st.button("🔮 Dự đoán loại hoa", type="primary", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction_proba = model.predict(input_scaled)
        predicted_class_idx = np.argmax(prediction_proba, axis=1)
        predicted_flower = label_encoder.inverse_transform(predicted_class_idx)[0]
        confidence = prediction_proba[0][predicted_class_idx[0]] * 100
        
        st.subheader("🎉 Kết quả dự đoán:")
        if predicted_flower == "Iris-setosa":
            st.success(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🔴")
        elif predicted_flower == "Iris-versicolor":
            st.info(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🟢")
        else:
            st.warning(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🔵")
            
        st.write(f"📊 Độ tin cậy: **{confidence:.2f}%**")
        
        # Bảng xác suất chi tiết
        proba_df = pd.DataFrame({
            'Loại hoa': label_encoder.classes_,
            'Xác suất (%)': [f"{p*100:.2f}%" for p in prediction_proba[0]]
        })
        st.table(proba_df)

with main_col2:
    st.subheader("📸 Hình ảnh minh họa & Kiểm tra")
    
    # Ô cho phép người dùng upload ảnh hoa của họ lên để demo
    uploaded_image = st.file_uploader("Tải ảnh bông hoa Iris của bạn lên đây (chỉ để hiển thị minh họa):", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Nếu người dùng upload ảnh, hiển thị ảnh của họ lên
        image = Image.open(uploaded_image)
        st.image(image, caption="Ảnh hoa bạn vừa tải lên", use_container_width=True)
    else:
        # Nếu chưa upload, hiển thị một ảnh hướng dẫn phân biệt Đài hoa (Sepal) và Cánh hoa (Petal) mặc định từ internet
        st.info("💡 Mẹo: Bạn có thể tải ảnh hoa lên đây cho giao diện thêm sinh động!")
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Petal-sepal.jpg", 
                 caption="Hình ảnh hướng dẫn: Phân biệt Đài hoa (Sepal) và Cánh hoa (Petal)", 
                 use_container_width=True)

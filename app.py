import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. Cấu hình giao diện Trang Web
st.set_page_config(
    page_title="Phân loại Hoa Iris - Nhóm 6",
    page_icon="🌸",
    layout="centered"
)

# 2. Hàm tải các file Model và Tiền xử lý (đã lưu từ Colab)
@st.cache_resource
def load_prediction_resources():
    # Tải mô hình ANN (.keras)
    model = load_model("ann_iris_model.keras")
    
    # Tải Scaler và Label Encoder (.pkl)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_prediction_resources()
except Exception as e:
    st.error(f"Không thể tải file model hoặc tiền xử lý. Hãy chắc chắn bạn đã upload đủ các file 'ann_iris_model.keras', 'scaler.pkl' và 'label_encoder.pkl' lên cùng thư mục GitHub. Chi tiết lỗi: {e}")
    st.stop()

# 3. Thiết kế Giao diện UI
st.title("🌸 Ứng Dụng Phân Loại Hoa Iris (ANN Nhiều Lớp)")
st.write("Sản phẩm được phát triển bởi **Nhóm 6**. Nhập thông số kích thước của hoa bên dưới để mô hình dự đoán chủng loại.")

st.markdown("---")
st.subheader("📥 Nhập thông số đặc trưng (cm):")

# Tạo 2 cột để giao diện nhập liệu cân đối hơn
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Chiều dài đài hoa (Sepal Length)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Chiều rộng đài hoa (Sepal Width)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Chiều dài cánh hoa (Petal Length)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Chiều rộng cánh hoa (Petal Width)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.markdown("---")

# 4. Logic Dự đoán khi bấm nút
if st.button("🔮 Dự đoán loại hoa", type="primary"):
    # Gộp các thông số thành mảng 2D giống cấu trúc đầu vào của bài mẫu
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # BƯỚC 1: Chuẩn hóa dữ liệu bằng Scaler đã học từ tập Train
    input_scaled = scaler.transform(input_data)
    
    # BƯỚC 2: Đưa qua mô hình ANN để dự đoán phân phối xác suất
    prediction_proba = model.predict(input_scaled)
    
    # BƯỚC 3: Lấy index có xác suất cao nhất
    predicted_class_idx = np.argmax(prediction_proba, axis=1)
    
    # BƯỚC 4: Giải mã ngược index thành tên loài hoa gốc (Iris-setosa, Iris-versicolor, Iris-virginica)
    predicted_flower = label_encoder.inverse_transform(predicted_class_idx)[0]
    confidence = prediction_proba[0][predicted_class_idx[0]] * 100
    
    # 5. Hiển thị kết quả ra giao diện
    st.subheader("🎉 Kết quả dự đoán:")
    
    # Tạo box thông báo kết quả sinh động kèm icon tương ứng
    if predicted_flower == "Iris-setosa":
        st.success(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🔴")
    elif predicted_flower == "Iris-versicolor":
        st.info(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🟢")
    else:
        st.warning(f"Mô hình dự đoán đây là loại hoa: **{predicted_flower}** 🔵")
        
    st.write(f"📊 Độ tin cậy của mô hình: **{confidence:.2f}%**")
    
    # Hiển thị bảng xác suất chi tiết cho cả 3 loài hoa để giao diện trực quan hơn
    st.markdown("##### Xác suất chi tiết của từng loại:")
    proba_df = pd.DataFrame({
        'Loại hoa': label_encoder.classes_,
        'Xác suất (%)': [f"{p*100:.2f}%" for p in prediction_proba[0]]
    })
    st.table(proba_df)

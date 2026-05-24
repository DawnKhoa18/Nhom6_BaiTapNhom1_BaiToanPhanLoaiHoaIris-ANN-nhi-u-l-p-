import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# 1. Cấu hình giao diện Trang Web
st.set_page_config(
    page_title="Phân loại Hoa Iris - Nhóm 6",
    page_icon="🌸",
    layout="wide"
)

# 2. Hàm tải các file Model và Tiền xử lý (đã lưu từ Colab)
@st.cache_resource
def load_prediction_resources():
    model = load_model("ann_iris_model.keras", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_prediction_resources()
except Exception as e:
    st.error(f"Lỗi tải file mô hình: {e}")
    st.stop()

# 3. Định nghĩa thư viện hình ảnh của 3 loài hoa để hiển thị sau khi dự đoán
HÌNH_ẢNH_HOA = {
    "Iris-setosa": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "mota": "Đặc trưng: Đài hoa rộng, cánh hoa rất nhỏ và ngắn."
    },
    "Iris-versicolor": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "mota": "Đặc trưng: Kích thước đài và cánh hoa ở mức trung bình, màu xanh tím."
    },
    "Iris-virginica": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "mota": "Đặc trưng: Cánh hoa rất lớn, dài và có vân đậm rõ rệt."
    }
}

# 4. Thiết kế Giao diện UI
st.title("🌸 Ứng Dụng Phân Loại Hoa Iris (ANN Nhiều Lớp)")
st.write("Sản phẩm được phát triển bởi **Nhóm 6**. Nhập thông số và xem kết quả trực quan dưới dạng hình ảnh.")
st.markdown("---")

# Chia màn hình làm 2 cột: Cột trái nhập số - Cột phải hiển thị kết quả & hình ảnh
col_trai, col_phai = st.columns([1, 1])

with col_trai:
    st.subheader("📥 Nhập thông số đặc trưng (cm):")
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        sepal_length = st.number_input("Chiều dài đài hoa (Sepal Length)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Chiều rộng đài hoa (Sepal Width)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
    with sub_col2:
        petal_length = st.number_input("Chiều dài cánh hoa (Petal Length)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
        petal_width = st.number_input("Chiều rộng cánh hoa (Petal Width)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
    
    st.markdown("---")
    
    # Kiểm tra biến trạng thái tồn tại chưa bằng cú pháp chuẩn 'not in st.session_state'
    if 'predicted_flower' not in st.session_state:
        st.session_state.predicted_flower = None
        st.session_state.confidence = 0.0
        st.session_state.prediction_proba = None

    # Xử lý logic khi bấm nút Dự đoán
    if st.button("🔮 Dự đoán loài hoa", type="primary", use_container_width=True):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        
        prediction_proba = model.predict(input_scaled)
        predicted_class_idx = np.argmax(prediction_proba, axis=1)
        
        st.session_state.predicted_flower = label_encoder.inverse_transform(predicted_class_idx)[0]
        st.session_state.confidence = prediction_proba[0][predicted_class_idx[0]] * 100
        st.session_state.prediction_proba = prediction_proba[0]

    # Hiển thị bảng xác suất nếu đã có kết quả dự đoán
    if st.session_state.predicted_flower is not None:
        st.subheader("📊 Xác suất phân loại chi tiết:")
        proba_df = pd.DataFrame({
            'Loại hoa': label_encoder.classes_,
            'Xác suất (%)': [f"{p*100:.2f}%" for p in st.session_state.prediction_proba]
        })
        st.table(proba_df)

with col_phai:
    st.subheader("🖥️ Kết quả kết xuất hình ảnh")
    
    if st.session_state.predicted_flower is None:
        st.info("💡 Điền thông số ở cột bên trái và bấm nút 'Dự đoán' để kích hoạt máy ảnh nhận diện!")
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Petal-sepal.jpg", 
                 caption="Sơ đồ cấu trúc Đài hoa (Sepal) và Cánh hoa (Petal)", 
                 use_container_width=True)
    else:
        hoa = st.session_state.predicted_flower
        do_tin_cay = st.session_state.confidence
        thong_tin_hoa = HÌNH_ẢNH_HOA[hoa]
        
        if hoa == "Iris-setosa":
            st.success(f"🎉 Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) 🔴")
        elif hoa == "Iris-versicolor":
            st.info(f"🎉 Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) 🟢")
        else:
            st.warning(f"🎉 Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) 🔵")
            
        st.image(thong_tin_hoa["url"], caption=f"Hình ảnh thực tế loài hoa {hoa}. {thong_tin_hoa['mota']}", use_container_width=True)

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Giao diện Web
st.set_page_config(
    page_title="Phân loại Hoa Iris - Nhóm 6",
    layout="wide"
)

# Hàm tải các file Model và dữ liệu đánh giá
@st.cache_resource
def load_all_resources():
    model = load_model("ann_iris_model.keras", compile=False)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
        
    # Tải thêm dữ liệu đánh giá mô hình
    with open("history_data.pkl", "rb") as f:
        history_dict = pickle.load(f)
    with open("test_metrics.pkl", "rb") as f:
        metrics_data = pickle.load(f)
        
    return model, scaler, label_encoder, history_dict, metrics_data

try:
    model, scaler, label_encoder, history_dict, metrics_data = load_all_resources()
except Exception as e:
    st.error(f"Lỗi tải. Hãy chắc chắn bạn đã upload đủ các file (.keras, .pkl) lên GitHub. Chi tiết: {e}")
    st.stop()

# Hình ảnh loài hoa 
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

# Tiêu đề
st.title("Phân Tích & Phân Loại Hoa Iris (ANN)")
st.subheader("Nhóm 6")
st.markdown("---")

# Chia tab giao diện
tab1, tab2 = st.tabs([":crystal_ball: Dự Đoán Trực Quan", ":bar_chart: Đánh Giá Hiệu Năng Mô Hình"])

# Tab dự đoán
with tab1:
    col_trai, col_phai = st.columns([1, 1])
    
    with col_trai:
        st.markdown("### Nhập thông số đặc trưng (cm):")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            sepal_length = st.number_input("Chiều dài đài hoa (Sepal Length)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
            sepal_width = st.number_input("Chiều rộng đài hoa (Sepal Width)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
        with sub_col2:
            petal_length = st.number_input("Chiều dài cánh hoa (Petal Length)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
            petal_width = st.number_input("Chiều rộng cánh hoa (Petal Width)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
        
        st.markdown("---")
        
        if 'pred_flower' not in st.session_state:
            st.session_state.pred_flower = None
            st.session_state.conf = 0.0
            st.session_state.prob = None

        if st.button(":crystal_ball: Tiến hành dự đoán", type="primary", use_container_width=True):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            prediction_proba = model.predict(input_scaled)
            predicted_class_idx = np.argmax(prediction_proba, axis=1)
            
            st.session_state.pred_flower = label_encoder.inverse_transform(predicted_class_idx)[0]
            st.session_state.conf = prediction_proba[0][predicted_class_idx[0]] * 100
            st.session_state.prob = prediction_proba[0]

        if st.session_state.pred_flower is not None:
            st.markdown("### :bar_chart: Xác suất phân loại chi tiết:")
            proba_df = pd.DataFrame({
                'Loại hoa': label_encoder.classes_,
                'Xác suất (%)': [f"{p*100:.2f}%" for p in st.session_state.prob]
            })
            st.table(proba_df)

    with col_phai:
        st.markdown("### :desktop_computer: Kết quả kết xuất hình ảnh")
        if st.session_state.pred_flower is None:
            st.info("Điền thông số ở cột bên trái và bấm nút 'Dự đoán' để kích hoạt máy ảnh nhận diện!")
            st.image("https://upload.wikimedia.org/wikipedia/commons/7/78/Petal-sepal.jpg", 
                     caption="Sơ đồ cấu trúc Đài hoa (Sepal) và Cánh hoa (Petal)", use_container_width=True)
        else:
            hoa = st.session_state.pred_flower
            do_tin_cay = st.session_state.conf
            thong_tin_hoa = HÌNH_ẢNH_HOA[hoa]
            
            if hoa == "Iris-setosa":
                st.success(f":tada: Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) :red_circle:")
            elif hoa == "Iris-versicolor":
                st.info(f":tada: Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) :green_circle:")
            else:
                st.warning(f":tada: Kết quả: **{hoa}** (Độ tin cậy: {do_tin_cay:.2f}%) :large_blue_circle:")
                
            st.image(thong_tin_hoa["url"], caption=f"Hình ảnh thực tế loài hoa {hoa}. {thong_tin_hoa['mota']}", use_container_width=True)


# Tab đánh giá mô hình
with tab2:
    st.markdown("## :chart_with_upwards_trend: Kết Quả Thực Nghiệm Mạng Neural Nhân Tạo (ANN)")
    st.write("Số liệu thu được trong quá trình huấn luyện và kiểm thử mô hình trên tập dữ liệu Iris.")
    
    # Hiển thị các ô chỉ số lớn
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Độ chính xác tập Test (Accuracy)", value=f"{metrics_data['test_accuracy']*100:.2f}%")
    with metric_col2:
        st.metric(label="Độ mất mát tập Test (Loss)", value=f"{metrics_data['test_loss']:.4f}")
    with metric_col3:
        st.metric(label="Thuật toán tối ưu", value="Adam")

    st.markdown("---")
    
    # Vẽ biểu đồ Accuracy & Loss qua các Epoch
    st.subheader(":chart_with_upwards_trend: Biểu đồ Quá trình Huấn luyện")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ Accuracy lịch sử
    ax1.plot(history_dict['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('Mô hình Accuracy qua các Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--')

    # Biểu đồ Loss lịch sử
    ax2.plot(history_dict['loss'], label='Train Loss', color='#d62728', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Validation Loss', color='#2ca02c', linewidth=2)
    ax2.set_title('Mô hình Loss qua các Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    
    st.pyplot(fig) 

    st.markdown("---")
    
    # Phân chia khu vực vẽ Confusion Matrix và Classification Report
    eval_col1, eval_col2 = st.columns([1.2, 1])
    
    with eval_col1:
        st.subheader(":jigsaw: Ma trận nhầm lẫn (Confusion Matrix)")
        
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(metrics_data['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax_cm)
        plt.xlabel('Nhãn dự đoán', fontsize=10, fontweight='bold')
        plt.ylabel('Nhãn thực tế', fontsize=10, fontweight='bold')
        plt.title('Confusion Matrix trên tập dữ liệu Test', fontsize=12, fontweight='bold')
        
        st.pyplot(fig_cm)

    with eval_col2:
        st.subheader(":clipboard: Báo cáo phân loại (Classification Report)")
        st.write("Chi tiết các chỉ số Precision, Recall, F1-score cho từng loài hoa:")
        
        report = classification_report(metrics_data['y_test'], metrics_data['y_pred'], target_names=label_encoder.classes_)
        st.code(report, language="text")
        
        st.info(":light_bulb: **Giải thích:**\n"
                "- **Precision:** Tỉ lệ dự đoán đúng trong những ca mô hình đoán là loài hoa đó.\n"
                "- **Recall:** Tỉ lệ tìm thấy đúng loài hoa đó trên tổng số hoa thực tế.\n"
                "- **F1-score:** Chỉ số trung bình điều hòa giữa Precision và Recall.")

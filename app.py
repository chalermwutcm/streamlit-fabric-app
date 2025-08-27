# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from fabric_data import FABRIC_INFO, CLASS_NAMES

# --- การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="แอพวิเคราะห์ลายผ้าไทย",
    page_icon="🎨",
    layout="wide"
)

# --- Caching Model: โหลดโมเดลเพียงครั้งเดียวเพื่อประสิทธิภาพที่ดีขึ้น ---
@st.cache_resource
def load_keras_model():
    """
    โหลดโมเดล Keras จากไฟล์ .h5
    """
    try:
        model = tf.keras.models.load_model('best_model_Custom_CNN.h5')
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None

model = load_keras_model()

# --- ฟังก์ชันประมวลผลภาพ ---
def preprocess_image(image):
    """
    เตรียมรูปภาพให้พร้อมสำหรับโมเดล (เหมือนกับใน p7.py)
    """
    # แปลง PIL Image เป็น Numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. ปรับขนาดภาพ
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # 2. ปรับสีและ Histogram Equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # 3. กรองสัญญาณรบกวน
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 4. เพิ่มความคมชัด
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # 5. Normalization และเพิ่ม Dimension
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- ส่วนของ UI ---
st.title("🎨 แอพพลิเคชันวิเคราะห์ลายผ้าไทย")
st.write(
    "อัปโหลดรูปภาพผ้าของคุณเพื่อให้ AI ช่วยวิเคราะห์ว่าเป็นลายอะไร พร้อมเรียนรู้ประวัติย่อที่น่าสนใจ"
)

uploaded_file = st.file_uploader(
    "เลือกรูปภาพ...", type=["jpg", "jpeg", "png"]
)

if model and uploaded_file is not None:
    # 1. แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่คุณอัปโหลด', use_column_width=True)

    # 2. ประมวลผลและทำนายผล
    with st.spinner('กำลังวิเคราะห์ลายผ้า...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        # 3. ดึงข้อมูลจาก fabric_data.py
        result_info = FABRIC_INFO.get(predicted_class_name)

    st.success(f"วิเคราะห์สำเร็จ!")

    # 4. แสดงผลลัพธ์
    if result_info:
        st.header(f"ลายผ้าคือ: {result_info['thai_name']}")
        st.subheader(f"ความมั่นใจ: {confidence:.2f}%")
        
        st.markdown("---")
        
        st.subheader("ประวัติย่อ")
        st.write(result_info['history'])

elif model is None:
    st.error("ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบไฟล์โมเดล")
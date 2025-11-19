import streamlit as st
import requests
from PIL import Image
import io

# Backend URL
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="CIFAR-3 Image Classifier",
    page_icon="🚀",
    layout="centered"
)

st.title("🔍 CIFAR-3 Image Classifier Demo")
st.write("Upload an image and the model will classify it as **airplane**, **automobile**, or **ship**.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    st.write("⏳ Predicting...")

    # Send request to backend
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(API_URL, files=files)
        
        if response.status_code != 200:
            st.error(f"Backend error: {response.text}")
        else:
            result = response.json()
            pred_class = result["pred_class"]
            probs = result["probs"]

            st.success(f"### 🧩 Prediction: **{pred_class.upper()}**")

            # Display probability bars
            st.write("### Probability Distribution")
            st.bar_chart([probs["airplane"], probs["automobile"], probs["ship"]], 
                         )
            st.write(f"Airplane: **{probs['airplane']:.3f}**")
            st.write(f"Automobile: **{probs['automobile']:.3f}**")
            st.write(f"Ship: **{probs['ship']:.3f}**")
    
    except Exception as e:
        st.error(f"Request failed: {e}")

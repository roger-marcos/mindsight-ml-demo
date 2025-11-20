import streamlit as st
import requests
from PIL import Image

# Backend URL (make sure this is the correct Render URL for the backend)
API_URL = "https://mindsight-ml-demo.onrender.com/predict"

# Set the Streamlit page config
st.set_page_config(
    page_title="CIFAR-3 Image Classifier",
    page_icon="🚀",
    layout="centered"
)

# Page Title
st.title("🔍 CIFAR-3 Image Classifier Demo")
st.write("Upload an image and the model will classify it as **airplane**, **automobile**, or **ship**.")

# File uploader to allow users to upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    # Indicate that the model is working on the prediction
    st.write("⏳ Predicting...")

    # Send the image to the backend via a POST request
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    
    try:
        # Make the POST request to the backend
        response = requests.post(API_URL, files=files)
        
        if response.status_code != 200:
            # If the backend returns an error, show it to the user
            st.error(f"Backend error: {response.text}")
        else:
            # If the request is successful, parse and display the results
            result = response.json()
            pred_class = result["pred_class"]
            probs = result["probs"]

            # Display the predicted class
            st.success(f"### 🧩 Prediction: **{pred_class.upper()}**")

            # Display the probability distribution for each class
            st.write("### Probability Distribution")
            st.bar_chart([probs["airplane"], probs["automobile"], probs["ship"]])

            # Display the individual class probabilities
            st.write(f"Airplane: **{probs['airplane']:.3f}**")
            st.write(f"Automobile: **{probs['automobile']:.3f}**")
            st.write(f"Ship: **{probs['ship']:.3f}**")
    
    except Exception as e:
        # If there is an error making the request, display it to the user
        st.error(f"Request failed: {e}")

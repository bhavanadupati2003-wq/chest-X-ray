# ==============================
# IMPORTS (MUST BE AT TOP)
# ==============================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ==============================
# CSS: ONLY LARGE BUTTON (NO BACKGROUND COLOR)
# ==============================
st.markdown("""
<style>
/* Make Predict button large */
div.stButton > button {
    width: 100% !important;
    height: 70px !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    margin-top: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.title("🫁 Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image to predict **NORMAL** or **PNEUMONIA**")

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)

    state_dict = torch.load(
        "resnet18_pneumonia_model.pth",
        map_location=device
    )

    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)

    model.load_state_dict(state_dict, strict=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.to(device)
    model.eval()
    return model

model = load_model()

# ==============================
# IMAGE TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["NORMAL", "PNEUMONIA"]

# ==============================
# FILE UPLOADER
# ==============================
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION
# ==============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

    predict = st.button("🔍 Predict")

    if predict:
        with st.spinner("Analyzing X-ray..."):
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            prediction = class_names[pred.item()]
            confidence = conf.item() * 100

        st.subheader("🧾 Prediction Result")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        if prediction == "PNEUMONIA":
            st.error("⚠️ Pneumonia detected. Please consult a medical professional.")
        else:
            st.success("✅ Lungs appear normal.")

# ==============================
# FOOTER
# ==============================
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ by Bhavana | Machine Learning Engineer</p>",
    unsafe_allow_html=True
)

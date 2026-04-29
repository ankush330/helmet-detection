import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


st.set_page_config(page_title="Helmet Detection", page_icon="")
st.title(" Helmet Detection")
st.write("Upload an image to check if the person is wearing a helmet.")


@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load('helmet_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
class_names = ['helmet', 'no_helmet']


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    return class_names[predicted.item()], confidence.item() * 100


uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence = predict(image)

    if label == 'helmet':
        st.success(f"✅ Helmet Detected! ({confidence:.1f}% confident)")
    else:
        st.error(f"❌ No Helmet! ({confidence:.1f}% confident)")
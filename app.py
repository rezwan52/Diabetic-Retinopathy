import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image
import timm
import numpy as np
import plotly.express as px
import cv2




# ---------------- CONFIG ----------------
IMG_SIZE = 304
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- HybridModel ----------------
class HybridModel(nn.Module):
    def __init__(self, freeze_backbones=False):
        super().__init__()
        # ResNet18
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            resnet = models.resnet18(pretrained=False)
        resnet.fc = nn.Identity(); res_dim = 512

        # InceptionV3
        try:
            inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        except Exception:
            inception = models.inception_v3(pretrained=False)
        inception.aux_logits = False
        inception.fc = nn.Identity(); inc_dim = 2048

        # ViT-Tiny
        try:
            vit = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0, img_size=IMG_SIZE)
        except Exception:
            vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=IMG_SIZE)
        vit_dim = vit.num_features

        self.resnet, self.inception, self.vit = resnet, inception, vit

        if freeze_backbones:
            for m in [self.resnet, self.inception, self.vit]:
                for p in m.parameters(): p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(res_dim + inc_dim + vit_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        r = self.resnet(x)
        i = self.inception(x)
        v = self.vit(x)
        z = torch.cat([r,i,v], dim=1)
        logit = self.head(z).squeeze(1)
        return logit

# ---------------- Model load ----------------
import streamlit as st
import torch
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """
    Load a PyTorch model with two checkpoints: best_aptos.pt and best_bd.pt
    """
    model = HybridModel().to(device)

    # Load first checkpoint (best_aptos)
    checkpoint_path1 = os.path.join("checkpoints", "best_aptos.pt")
    if not os.path.exists(checkpoint_path1):
        st.error(f"Checkpoint not found: {checkpoint_path1}")
        st.stop()
    checkpoint1 = torch.load(checkpoint_path1, map_location=device)
    if isinstance(checkpoint1, dict) and "state_dict" in checkpoint1:
        model.load_state_dict(checkpoint1["state_dict"])
    elif isinstance(checkpoint1, dict):
        model.load_state_dict(checkpoint1)

    # Load second checkpoint (best_bd)
    checkpoint_path2 = os.path.join("checkpoints", "best_bd.pt")
    if not os.path.exists(checkpoint_path2):
        st.error(f"Checkpoint not found: {checkpoint_path2}")
        st.stop()
    checkpoint2 = torch.load(checkpoint_path2, map_location=device)
    if isinstance(checkpoint2, dict) and "state_dict" in checkpoint2:
        model.load_state_dict(checkpoint2["state_dict"])
    elif isinstance(checkpoint2, dict):
        model.load_state_dict(checkpoint2)

    model.eval()
    return model


# ---------------- Image preprocessing ----------------
def preprocess_image(image):
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(image).unsqueeze(0).to(device)

# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.tlayer = target_layer
        self.grads = None; self.acts = None
        self.fh = self.tlayer.register_forward_hook(self._fwd)
        self.bh = self.tlayer.register_full_backward_hook(self._bwd)
    def _fwd(self, m, i, o): self.acts = o
    def _bwd(self, m, gi, go): self.grads = go[0]
    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = torch.sigmoid(logits).mean()
        score.backward(retain_graph=True)
        w = self.grads.mean(dim=(2,3), keepdim=True)
        cam = (self.acts * w).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu()
    def close(self):
        self.fh.remove(); self.bh.remove()

# ---------------- Interactive overlay ----------------
def interactive_gradcam(model, image):
    input_tensor = preprocess_image(image)
    target_layer = model.resnet.layer4[-1].conv2
    cam_engine = GradCAM(model, target_layer)
    input_tensor.requires_grad_(True)
    cam = cam_engine(input_tensor).squeeze().cpu().numpy()
    cam_engine.close()

    # Resize cam to original image
    cam_resized = cv2.resize(cam, (image.width, image.height))
    fig = px.imshow(cam_resized, color_continuous_scale="Jet", origin="upper")
    fig.update_layout(coloraxis_colorbar=dict(title="Attention"), margin=dict(l=0,r=0,t=0,b=0))
    return fig, cam_resized

# ---------------- Streamlit UI ----------------
st.title("🩺 Diabetic Retinopathy Detection With Explainable-AI")

# Model selection
model_choice = st.radio("Select Model", ["APTOS 2019", "Bangladeshi DR"])
model_type = "aptos" if model_choice=="APTOS 2019" else "bd"

uploaded_file = st.file_uploader("Upload a retina image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = (prob >= 0.5)

     # Display prediction + confidence
    if pred == 0:
        healthy_conf = (1 - prob) * 100
        st.success(f"✅ No Diabetic Retinopathy detected (Confidence: {healthy_conf:.2f}%)")
        st.markdown(f"""
        **Explanation:** মডেল বলছে retina তে কোন DR নেই।  
        Confidence মানে, model কতটা নিশ্চিত যে retina তে DR নেই।  
        DR probability: {prob*100:.2f}% → Healthy confidence: {healthy_conf:.2f}%
        """)
    else:
        dr_conf = prob * 100
        st.error(f"⚠️ Diabetic Retinopathy detected (Confidence: {dr_conf:.2f}%)")
        st.markdown(f"""
        **Explanation:** মডেল বলছে retina তে Diabetic Retinopathy আছে।  
        Confidence মানে, model কতটা নিশ্চিত যে retina তে DR আছে।  
        DR probability: {dr_conf:.2f}%
        """)

    # Interactive Grad-CAM
    st.subheader("Grad-CAM")
    fig, cam_resized = interactive_gradcam(model, image)
    st.plotly_chart(fig, use_container_width=True)

    # Hover explanation
    st.markdown("""
    **Hover Explanation:**  
    - 🔴 High attention: model DR detect করার জন্য সবচেয়ে বেশি focus করেছে  
    - 🟠 Medium attention: model moderate focus  
    - 🟢 Low attention: model কম focus করেছে  
    """)

     # ---------------- Health Advice ----------------
    st.subheader("Medical Advice / Next Steps")
    if pred == 1:
        st.warning("""
        মডেল বলছে DR detect হয়েছে।  
        ✅ পরবর্তী পদক্ষেপ:
        1. Retina specialist দেখানো
        2. Regular eye check-up
        3. Blood sugar ও BP control
        4. Doctor নির্দেশ অনুযায়ী medication / laser / injection
        5. Healthy lifestyle বজায় রাখা
        """)
    else:
        st.info("""
        মডেল বলছে DR detect হয়নি।  
        ✅ Preventive measures:
        1. Diabetes থাকলে yearly eye check-up
        2. Sugar, BP, cholesterol control
        3. Balanced diet + exercise
        4. Smoking & alcohol limited
        """)

    # ---------------- Grad-CAM explanation ----------------
    st.subheader("Prediction Explanation")
    if pred == 1:
        st.markdown("""
        মডেল DR detect করেছে।  
        Grad-CAM overlay এ লাল অংশগুলো দেখাচ্ছে retina তে যেসব region model সবচেয়ে বেশি focus করেছে।  
        এই region গুলোতে অস্বাভাবিক blood vessels, microaneurysms বা hemorrhages থাকতে পারে।  
        অর্থাৎ মডেল বলছে এই অংশের কারণে DR detect হয়েছে।  
        Color intensity (>0.7 high, 0.3-0.7 medium, <0.3 low) ব্যবহার করে user বুঝতে পারবে কোন region বেশি গুরুত্বপূর্ণ।
        """)
    else:
        st.markdown("""
        মডেল বলছে DR detect হয়নি।  
        Grad-CAM overlay এ কোনো prominent hotspot নেই।  
        Model normal retina texture এবং vessels pattern দেখে healthy verdict দিয়েছে।  
        Color intensity hints দেখাবে model কোথায় বেশি বা কম focus করেছে।
        """)

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image
import timm

# ---------- CONFIG ----------
IMG_SIZE = 304
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- HybridModel ----------
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

# ---------- Model load ----------
@st.cache_resource
def load_model():
    model = HybridModel().to(device)
    checkpoint = torch.load("checkpoints/best_aptos.pt", map_location=device)
    # checkpoint dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# ---------- Image preprocessing ----------
def preprocess_image(image):
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(image).unsqueeze(0).to(device)

# ---------- Streamlit UI ----------
st.title("🩺 Diabetic Retinopathy Detection (APTOS)")

uploaded_file = st.file_uploader("Upload a retina image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        pred = (torch.sigmoid(output) >= 0.5).long().item()
        prob = torch.sigmoid(output).item()

    if pred == 0:
        st.success(f"✅ No Diabetic Retinopathy detected (Confidence: {prob*100:.2f}%)")
    else:
        st.error(f"⚠️ Diabetic Retinopathy detected (Confidence: {prob*100:.2f}%)")

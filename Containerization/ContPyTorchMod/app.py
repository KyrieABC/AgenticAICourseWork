from fastapi import  UploadFile, FastAPI
from PIL import Image
import torch
import torchvision.transforms as T

app = FastAPI()

model = torch.load("model.pt",map_location="cpu")
model.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(file.file).convert("RGB")
    x=transform(img).unsqueeze(0)
    with torch.no_grad():
        y=model(x)
    pred= y.argmax().item()
    return {"prediction": int(pred)}

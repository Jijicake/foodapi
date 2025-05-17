from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_image
from PIL import Image
import io

app = FastAPI()

# Optional CORS if testing from mobile app locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_endpoint(
    file: UploadFile = File(...),
    cut_back: str = Form(""),         # comma-separated e.g., "sugar,salt"
    penalty: str = Form("")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    cut_back_list = [i.strip() for i in cut_back.split(",")] if cut_back else []
    result = analyze_image(image, cut_back=cut_back_list, penalty_override=penalty)
    return result

import os
import base64
import io
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from torchvision import transforms

from ..models.alexnet import ModernAlexNet
from ..utils.gradcam import GradCAM

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def create_app(config: dict) -> FastAPI:
    paths_cfg = config["paths"]

    data_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("data_dir", "./data")))
    curated_dir = os.path.join(data_dir, "curated")
    weights_dir = os.path.abspath(os.path.expanduser(paths_cfg.get("weights_dir", "./weights")))

    os.makedirs(curated_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    app = FastAPI(title="Modern AlexNet Pipeline Studio")

    # Static assets and template setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")
    app.mount("/media", StaticFiles(directory=curated_dir), name="media")
    templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

    # Model Cache
    loaded_model_cache = {}

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request=request, name="inference.html", context={"active_page": "inference"})

    @app.get("/curate", response_class=HTMLResponse)
    async def curate_page(request: Request):
        return templates.TemplateResponse(request=request, name="curate.html", context={"active_page": "curate"})

    @app.get("/inference", response_class=HTMLResponse)
    async def inference_page(request: Request):
        return templates.TemplateResponse(request=request, name="inference.html", context={"active_page": "inference"})

    # --- CURATION API ENDPOINTS ---

    @app.post("/api/curate/upload")
    async def upload_images(
        class_name: str = Form(...),
        image_size: int = Form(224),
        files: List[UploadFile] = File(...),
    ):
        clean_class = "".join(c for c in class_name if c.isalnum() or c in ("_", "-")).strip().lower()
        if not clean_class:
            raise HTTPException(status_code=400, detail="Invalid class name.")

        target_class_dir = os.path.join(curated_dir, clean_class)
        os.makedirs(target_class_dir, exist_ok=True)

        saved_count = 0
        for upload in files:
            ext = os.path.splitext(upload.filename.lower())[1]
            if ext not in VALID_EXTENSIONS:
                continue

            contents = await upload.read()
            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
                if image_size > 0:
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                idx = len(os.listdir(target_class_dir)) + 1
                out_name = f"img_{idx:05d}{ext if ext else '.png'}"
                out_path = os.path.join(target_class_dir, out_name)
                img.save(out_path)
                saved_count += 1
            except Exception as e:
                print(f"[!] Error processing image {upload.filename}: {e}")

        return JSONResponse({"status": "success", "count": saved_count, "class_name": clean_class})

    @app.get("/api/curate/stats")
    async def get_curate_stats():
        subdirs = sorted([d for d in os.listdir(curated_dir) if os.path.isdir(os.path.join(curated_dir, d))])
        classes_info = []
        total_imgs = 0

        for d in subdirs:
            p = os.path.join(curated_dir, d)
            count = len([f for f in os.listdir(p) if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS])
            classes_info.append({"name": d, "count": count})
            total_imgs += count

        return JSONResponse({"total_images": total_imgs, "classes": classes_info})

    @app.get("/api/curate/gallery")
    async def get_gallery(class_name: str = "all"):
        images = []
        subdirs = sorted([d for d in os.listdir(curated_dir) if os.path.isdir(os.path.join(curated_dir, d))])
        
        if class_name != "all" and class_name in subdirs:
            subdirs = [class_name]

        for d in subdirs:
            p = os.path.join(curated_dir, d)
            files = sorted([f for f in os.listdir(p) if os.path.splitext(f.lower())[1] in VALID_EXTENSIONS])
            for f in files[:30]:  # Limit gallery output to 30 items per class
                images.append({
                    "class_name": d,
                    "filename": f,
                    "url": f"/media/{d}/{f}"
                })

        return JSONResponse({"images": images})

    # --- INFERENCE API ENDPOINTS ---

    @app.get("/api/inference/models")
    async def list_models():
        if not os.path.exists(weights_dir):
            return JSONResponse({"models": []})

        files = sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt")])
        models_info = []
        for f in files:
            path = os.path.join(weights_dir, f)
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            models_info.append({"filename": f, "name": f, "size_mb": size_mb})

        return JSONResponse({"models": models_info})

    @app.post("/api/inference/predict")
    async def predict(
        model_filename: str = Form(...),
        file: UploadFile = File(...),
        enable_gradcam: bool = Form(False),
    ):
        model_path = os.path.join(weights_dir, model_filename)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model checkpoint '{model_filename}' not found.")

        # Load checkpoint & model state
        if model_filename not in loaded_model_cache:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            class_names = checkpoint.get("class_names", ["Class 0", "Class 1"])
            
            model = ModernAlexNet(
                num_classes=len(class_names),
                in_channels=checkpoint.get("config", {}).get("model", {}).get("in_channels", 3),
                use_batch_norm=checkpoint.get("config", {}).get("model", {}).get("use_batch_norm", True),
                dropout=checkpoint.get("config", {}).get("model", {}).get("dropout", 0.5),
            ).to(device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            loaded_model_cache[model_filename] = {
                "model": model,
                "class_names": class_names,
                "device": device,
            }

        cached = loaded_model_cache[model_filename]
        model = cached["model"]
        class_names = cached["class_names"]
        device = cached["device"]

        # Read image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Transform image
        img_size = config.get("dataset", {}).get("image_size", 224)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)[0]

        top_k = min(5, len(class_names))
        top_probs, top_indices = torch.topk(probs, top_k)

        predictions = []
        for i in range(top_k):
            idx = top_indices[i].item()
            prob = top_probs[i].item() * 100.0
            predictions.append({
                "class_name": class_names[idx] if idx < len(class_names) else f"Class {idx}",
                "probability": round(prob, 2),
            })

        # Grad-CAM heatmap generation if requested
        gradcam_b64 = None
        if enable_gradcam:
            try:
                cam = GradCAM(model)
                top_class_idx = top_indices[0].item()
                heatmap = cam.generate_heatmap(input_tensor, target_class=top_class_idx)
                resized_pil = pil_img.resize((img_size, img_size))
                overlay = cam.overlay_heatmap(resized_pil, heatmap)
                
                buf = io.BytesIO()
                overlay.save(buf, format="JPEG")
                gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"[!] Grad-CAM failed: {e}")

        return JSONResponse({
            "status": "success",
            "predictions": predictions,
            "gradcam_image_base64": gradcam_b64,
        })

    return app

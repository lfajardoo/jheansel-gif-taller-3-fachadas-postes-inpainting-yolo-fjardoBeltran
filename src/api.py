from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO

from utils import (
    ensure_dir,
    load_image,
    save_image,
    draw_boxes,
    build_mask_from_boxes,
    dilate_mask,
    apply_inpainting,
    filter_detections_by_class,
    result_to_detections,
)

app = FastAPI(title="API deteccion de fachadas y eliminacion de postes")

MODEL_PATH = "models/casas_postes_v1/weights/best.pt"

UPLOAD_DIR = "outputs/api_uploads"
PRED_DIR = "outputs/predicciones"
MASK_DIR = "outputs/mascaras_postes"
INPAINT_DIR = "outputs/resultados_inpainting"

ensure_dir(UPLOAD_DIR)
ensure_dir(PRED_DIR)
ensure_dir(MASK_DIR)
ensure_dir(INPAINT_DIR)

model = YOLO(MODEL_PATH)
class_names = model.names


def save_temp_file(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix if upload.filename else ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        return tmp.name


@app.get("/")
def root():
    return {
        "message": "API deteccion de fachadas y eliminacion de postes",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(file: UploadFile = File(...), conf: float = 0.25):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibio archivo")

    temp_path = save_temp_file(file)
    image = load_image(temp_path)

    results = model.predict(source=temp_path, conf=conf, verbose=False)
    result = results[0]
    detections = result_to_detections(result)

    output_path = str(Path(PRED_DIR) / f"{Path(file.filename).stem}_pred.jpg")
    pred_image = draw_boxes(image, detections, class_names)
    save_image(output_path, pred_image)

    return JSONResponse(
        {
            "filename": file.filename,
            "detections": detections,
            "prediction_image": output_path,
        }
    )


@app.post("/mask")
def mask(
    file: UploadFile = File(...),
    conf: float = 0.25,
    target_class: str = "poste",
    mask_kernel: int = 7,
    mask_iterations: int = 1,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibio archivo")

    temp_path = save_temp_file(file)
    image = load_image(temp_path)

    results = model.predict(source=temp_path, conf=conf, verbose=False)
    result = results[0]
    detections = result_to_detections(result)

    postes = filter_detections_by_class(
        detections=detections,
        target_class_name=target_class,
        class_names=class_names,
    )

    mask_img = build_mask_from_boxes(image.shape, postes)
    mask_img = dilate_mask(mask_img, kernel_size=mask_kernel, iterations=mask_iterations)

    output_path = str(Path(MASK_DIR) / f"{Path(file.filename).stem}_mask.png")
    save_image(output_path, mask_img)

    return JSONResponse(
        {
            "filename": file.filename,
            "target_class": target_class,
            "num_target_detections": len(postes),
            "mask_path": output_path,
        }
    )


@app.post("/inpaint")
def inpaint(
    file: UploadFile = File(...),
    conf: float = 0.25,
    target_class: str = "poste",
    mask_kernel: int = 7,
    mask_iterations: int = 1,
    inpaint_radius: int = 3,
    inpaint_method: str = "telea",
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibio archivo")

    temp_path = save_temp_file(file)
    image = load_image(temp_path)

    results = model.predict(source=temp_path, conf=conf, verbose=False)
    result = results[0]
    detections = result_to_detections(result)

    postes = filter_detections_by_class(
        detections=detections,
        target_class_name=target_class,
        class_names=class_names,
    )

    if not postes:
        raise HTTPException(status_code=404, detail=f"No se detecto la clase '{target_class}'")

    mask_img = build_mask_from_boxes(image.shape, postes)
    mask_img = dilate_mask(mask_img, kernel_size=mask_kernel, iterations=mask_iterations)

    output_path = str(Path(INPAINT_DIR) / f"{Path(file.filename).stem}_inpaint.jpg")
    result_img = apply_inpainting(
        image=image,
        mask=mask_img,
        radius=inpaint_radius,
        method=inpaint_method,
    )
    save_image(output_path, result_img)

    return JSONResponse(
        {
            "filename": file.filename,
            "target_class": target_class,
            "num_target_detections": len(postes),
            "inpaint_path": output_path,
        }
    )


@app.post("/inpaint/file")
def inpaint_file(file: UploadFile = File(...), conf: float = 0.25):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se recibio archivo")

    temp_path = save_temp_file(file)
    image = load_image(temp_path)

    results = model.predict(source=temp_path, conf=conf, verbose=False)
    result = results[0]
    detections = result_to_detections(result)

    postes = filter_detections_by_class(
        detections=detections,
        target_class_name="poste",
        class_names=class_names,
    )

    if not postes:
        raise HTTPException(status_code=404, detail="No se detectaron postes")

    mask_img = build_mask_from_boxes(image.shape, postes)
    mask_img = dilate_mask(mask_img, kernel_size=7, iterations=1)

    output_path = str(Path(INPAINT_DIR) / f"{Path(file.filename).stem}_inpaint.jpg")
    result_img = apply_inpainting(image=image, mask=mask_img, radius=3, method="telea")
    save_image(output_path, result_img)

    return FileResponse(output_path, media_type="image/jpeg", filename=Path(output_path).name)

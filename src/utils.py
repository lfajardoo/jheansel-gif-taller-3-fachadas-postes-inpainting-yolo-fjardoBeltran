import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    """Crea un directorio si no existe."""
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(source: str) -> List[str]:
    """
    Devuelve una lista de imágenes a procesar.
    Si source es archivo, devuelve [source].
    Si source es carpeta, devuelve todas las imágenes compatibles.
    """
    source_path = Path(source)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source_path.is_file():
        return [str(source_path)]

    if source_path.is_dir():
        images = [
            str(p) for p in sorted(source_path.iterdir())
            if p.suffix.lower() in valid_ext
        ]
        return images

    raise FileNotFoundError(f"No se encontró la ruta: {source}")


def load_image(image_path: str) -> np.ndarray:
    """Carga una imagen en formato BGR."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return image


def save_image(image_path: str, image: np.ndarray) -> None:
    """Guarda una imagen en disco."""
    parent = Path(image_path).parent
    ensure_dir(str(parent))
    ok = cv2.imwrite(image_path, image)
    if not ok:
        raise ValueError(f"No se pudo guardar la imagen en: {image_path}")


def draw_boxes(
    image: np.ndarray,
    detections: List[dict],
    class_names: dict
) -> np.ndarray:
    """
    Dibuja bounding boxes y etiquetas sobre la imagen.
    Cada detección debe tener:
        {
            "xyxy": [x1, y1, x2, y2],
            "cls_id": int,
            "conf": float
        }
    """
    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        cls_id = int(det["cls_id"])
        conf = float(det["conf"])
        label = class_names.get(cls_id, str(cls_id))

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(
            output,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def build_mask_from_boxes(
    image_shape: Tuple[int, int, int],
    detections: List[dict]
) -> np.ndarray:
    """
    Construye una máscara binaria a partir de bounding boxes.
    La máscara tendrá valor 255 en las zonas a eliminar.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = 7, iterations: int = 1) -> np.ndarray:
    """Dilata una máscara para cubrir mejor el objeto a remover."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_inpainting(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea"
) -> np.ndarray:
    """
    Aplica inpainting usando OpenCV.
    method:
        - 'telea'
        - 'ns'  (Navier-Stokes)
    """
    if method.lower() == "telea":
        flags = cv2.INPAINT_TELEA
    elif method.lower() == "ns":
        flags = cv2.INPAINT_NS
    else:
        raise ValueError("method debe ser 'telea' o 'ns'")

    result = cv2.inpaint(image, mask, radius, flags)
    return result


def filter_detections_by_class(
    detections: List[dict],
    target_class_name: str,
    class_names: dict
) -> List[dict]:
    """Filtra detecciones por nombre de clase."""
    filtered = []
    target_class_name = target_class_name.lower().strip()

    for det in detections:
        cls_id = int(det["cls_id"])
        cls_name = class_names.get(cls_id, str(cls_id)).lower().strip()
        if cls_name == target_class_name:
            filtered.append(det)

    return filtered


def result_to_detections(result) -> List[dict]:
    """
    Convierte un resultado de Ultralytics YOLO a lista de detecciones.
    """
    detections = []

    if result.boxes is None:
        return detections

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for box, score, class_id in zip(xyxy, conf, cls):
        detections.append(
            {
                "xyxy": box.tolist(),
                "conf": float(score),
                "cls_id": int(class_id),
            }
        )

    return detections


def get_output_paths(
    image_path: str,
    pred_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    inpaint_dir: Optional[str] = None,
):
    """
    Construye rutas de salida basadas en el nombre de la imagen.
    """
    stem = Path(image_path).stem

    pred_path = str(Path(pred_dir) / f"{stem}_pred.jpg") if pred_dir else None
    mask_path = str(Path(mask_dir) / f"{stem}_mask.png") if mask_dir else None
    inpaint_path = str(Path(inpaint_dir) / f"{stem}_inpaint.jpg") if inpaint_dir else None

    return pred_path, mask_path, inpaint_path

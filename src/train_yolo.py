import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de YOLOv8 para detección de casas y postes"
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo base YOLO")
    parser.add_argument("--data", type=str, default="data.yaml", help="Archivo data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="cpu, 0, 0,1, etc.")
    parser.add_argument("--project", type=str, default="models", help="Carpeta de salida")
    parser.add_argument("--name", type=str, default="casas_postes_v1", help="Nombre del experimento")
    parser.add_argument("--exist-ok", action="store_true", help="Permite sobrescribir salida")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Cargando modelo...")
    model = YOLO(args.model)

    print("Iniciando entrenamiento...")
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "plots": True,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    print("\nEntrenamiento finalizado.")
    print(f"Resultados guardados en: {results.save_dir}")

    print("\nEjecutando validación final...")
    metrics = model.val(data=args.data)

    print("Validación final completada.")
    print(metrics)


if __name__ == "__main__":
    main()
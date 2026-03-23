import argparse
from pathlib import Path

from ultralytics import YOLO

from utils import (
    ensure_dir,
    list_images,
    load_image,
    save_image,
    draw_boxes,
    build_mask_from_boxes,
    dilate_mask,
    apply_inpainting,
    filter_detections_by_class,
    result_to_detections,
    get_output_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia, generación de máscaras e inpainting para postes"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/casas_postes_v1/weights/best.pt",
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Ruta a una imagen o a una carpeta con imágenes"
    )
    parser.add_argument(
        "--modo",
        type=str,
        default="pipeline",
        choices=["detectar", "mascaras", "inpaint", "pipeline"],
        help="Modo de ejecución"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Umbral de confianza"
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="poste",
        help="Clase objetivo para generar máscara e inpainting"
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default="outputs/predicciones",
        help="Directorio para guardar imágenes con detecciones"
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="outputs/mascaras_postes",
        help="Directorio para guardar máscaras"
    )
    parser.add_argument(
        "--inpaint-dir",
        type=str,
        default="outputs/resultados_inpainting",
        help="Directorio para guardar resultados de inpainting"
    )
    parser.add_argument(
        "--mask-kernel",
        type=int,
        default=7,
        help="Tamaño del kernel para dilatación de máscara"
    )
    parser.add_argument(
        "--mask-iterations",
        type=int,
        default=1,
        help="Número de iteraciones de dilatación"
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=3,
        help="Radio del inpainting"
    )
    parser.add_argument(
        "--inpaint-method",
        type=str,
        default="telea",
        choices=["telea", "ns"],
        help="Método de inpainting de OpenCV"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(args.pred_dir)
    ensure_dir(args.mask_dir)
    ensure_dir(args.inpaint_dir)

    print("Cargando modelo...")
    model = YOLO(args.weights)
    class_names = model.names

    image_paths = list_images(args.source)

    print(f"Se encontraron {len(image_paths)} imagen(es) para procesar.")

    for image_path in image_paths:
        print(f"\nProcesando: {image_path}")

        image = load_image(image_path)
        results = model.predict(source=image_path, conf=args.conf, verbose=False)

        if not results:
            print("No se obtuvieron resultados.")
            continue

        result = results[0]
        detections = result_to_detections(result)

        pred_path, mask_path, inpaint_path = get_output_paths(
            image_path=image_path,
            pred_dir=args.pred_dir,
            mask_dir=args.mask_dir,
            inpaint_dir=args.inpaint_dir,
        )

        # 1. Detección
        if args.modo in ["detectar", "pipeline"]:
            image_with_boxes = draw_boxes(image, detections, class_names)
            save_image(pred_path, image_with_boxes)
            print(f"Predicción guardada en: {pred_path}")

        # 2. Máscara de postes
        postes = filter_detections_by_class(
            detections=detections,
            target_class_name=args.target_class,
            class_names=class_names,
        )

        if args.modo in ["mascaras", "pipeline", "inpaint"]:
            mask = build_mask_from_boxes(image.shape, postes)
            mask = dilate_mask(
                mask,
                kernel_size=args.mask_kernel,
                iterations=args.mask_iterations,
            )
            save_image(mask_path, mask)
            print(f"Máscara guardada en: {mask_path}")

        # 3. Inpainting
        if args.modo in ["inpaint", "pipeline"]:
            if len(postes) == 0:
                print("No se detectaron postes. Se omite inpainting.")
            else:
                inpainted = apply_inpainting(
                    image=image,
                    mask=mask,
                    radius=args.inpaint_radius,
                    method=args.inpaint_method,
                )
                save_image(inpaint_path, inpainted)
                print(f"Resultado de inpainting guardado en: {inpaint_path}")


if __name__ == "__main__":
    main()
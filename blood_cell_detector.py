import os
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'images_dir': 'Complete-Blood-Cell-Count-Dataset-master/Training/Images',
    'annotations_dir': 'Complete-Blood-Cell-Count-Dataset-master/Training/Annotations',
    'model_path': 'keremberke/yolov8n-blood-cell-detection',
    'num_samples': 5,
    'image_extensions': ['.jpg', '.jpeg', '.png'],
    'confidence_threshold': 0.25,
    'device': 'cpu'
}


def setup_pytorch_compatibility():
    import torch
    import ultralytics.nn.tasks as tasks

    original_torch_load = torch.load

    def patched_torch_load(f, map_location=None, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(f, map_location=map_location, **kwargs)

    tasks.torch.load = patched_torch_load
    logger.info("PyTorch compatibility fix applied")


def validate_directories(images_dir, annotations_dir):
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")


def get_image_files(images_dir, extensions, limit=None):
    all_images = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if any(f.lower().endswith(ext) for ext in extensions)
    ]
    if not all_images:
        raise ValueError(f"No images found in {images_dir}")
    selected = all_images[:limit] if limit else all_images
    logger.info(f"Found {len(all_images)} images, selected {len(selected)}")
    return selected


class BloodCellDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        from ultralyticsplus import YOLO
        self.model = YOLO(self.config['model_path'])
        logger.info(f"Model loaded: {self.config['model_path']}")

    def predict(self, image_path):
        result = self.model.predict(image_path, conf=self.config['confidence_threshold'])
        logger.info(f"Processed: {os.path.basename(image_path)}")
        return result

    def predict_batch(self, image_paths):
        results = []
        for img_path in image_paths:
            try:
                results.append(self.predict(img_path))
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
        logger.info(f"Processed {len(results)}/{len(image_paths)} images")
        return results

    def count_cells(self, results):
        counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0, 'Total': 0}
        for result_list in results:
            items = result_list if isinstance(result_list, list) else [result_list]
            for result in items:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                for box in result.boxes:
                    class_name = result.names[int(box.cls)]
                    if class_name == 'Platelets':
                        class_name = 'Platelet'
                    if class_name in counts:
                        counts[class_name] += 1
                        counts['Total'] += 1
        return counts

    def visualize_results(self, image_path, result, save_path=None):
        import cv2
        rendered = result[0].plot() if hasattr(result[0], 'plot') else cv2.imread(image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(image_path))
        plt.axis('off')
        if save_path:
            cv2.imwrite(save_path, rendered)
            logger.info(f"Saved visualization to {save_path}")
        plt.show()


def main():
    setup_pytorch_compatibility()
    validate_directories(CONFIG['images_dir'], CONFIG['annotations_dir'])

    image_files = get_image_files(CONFIG['images_dir'], CONFIG['image_extensions'], CONFIG['num_samples'])

    detector = BloodCellDetector(CONFIG)
    results = detector.predict_batch(image_files)
    counts = detector.count_cells(results)

    print(f"\n📊 Blood Cell Counts:")
    print(f"   RBC:      {counts['RBC']}")
    print(f"   WBC:      {counts['WBC']}")
    print(f"   Platelets:{counts['Platelet']}")
    print(f"   Total:    {counts['Total']}")

    if results:
        os.makedirs("predicted_outputs", exist_ok=True)
        detector.visualize_results(image_files[0], results[0], "predicted_outputs/result_visualization.jpg")


if __name__ == "__main__":
    main()

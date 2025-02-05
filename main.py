import logging
from os import path
from deepface import DeepFace


DATA_PATH = path.join(path.dirname(__file__), "data")

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting DeepFace script...")

    detections = DeepFace.analyze(
        img_path = path.join(DATA_PATH, "picture.jpg"),
        actions = ["age", "gender", "emotion"]
    )

    formatted_detections = [
        {
            "age": detection["age"],
            "gender": detection["dominant_gender"],
            "emotion": detection["dominant_emotion"],
            "region": {
                "x": detection["region"]["x"],
                "y": detection["region"]["y"],
                "width": detection["region"]["w"],
                "height": detection["region"]["h"],
            }
        }
        for detection in detections
    ]

    print(formatted_detections)

    logger.info("DeepFace script finished.")


if __name__ == "__main__":
    main()


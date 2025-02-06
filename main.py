import cv2
import logging
from deepface import DeepFace
from os import environ, path
from tqdm import tqdm

ANALYZE_ACTIONS = ["age", "gender", "emotion"]
DATA_PATH = path.join(path.dirname(__file__), "data")

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting DeepFace script...")

    logger.info("Getting video frames...")
    frames = extract_frames(path.join(DATA_PATH, "video.mp4"))

    frame_paths = []
    for index, frame in enumerate(frames):
        frame_path = path.join(DATA_PATH, "frames", f"{index}.jpg")

        cv2.imwrite(frame_path, frame)

        frame_paths.append(frame_path)

    logger.info(f"{len(frame_paths)} frame(s) found.")

    logger.info("Analyzing video...")
    video_analysis = {index: analyze_image(frame_path) for index, frame_path in enumerate(tqdm(frame_paths))}

    formatted_video_analysis = {
        key: [format_detection(detection) for detection in detections]
        for key, detections in video_analysis.items()
    }

    logger.info("Video analyzed.")

    logger.info("DeepFace script finished.")


def analyze_image(image_path):
    try:
        return DeepFace.analyze(img_path=image_path, actions=ANALYZE_ACTIONS)
    except:
        return []


def format_detection(detection):
    return {
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


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)

    frames = []
    success, frame = video.read()

    while success:
        frames.append(frame)
        success, frame = video.read()

    video.release()

    return frames


if __name__ == "__main__":
    main()


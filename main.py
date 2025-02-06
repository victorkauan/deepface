import os
import cv2
import logging
from deepface import DeepFace

ANALYZE_ACTIONS = ["age", "gender", "emotion"]
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
FRAMES_PATH = os.path.join(DATA_PATH, "frames")
OUTPUT_PATH = os.path.join(DATA_PATH, "output")

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting DeepFace script...")

    logger.info("Extracting video frames...")
    video_path = os.path.join(DATA_PATH, "video.mp4")
    frames = extract_frames(video_path)

    frame_paths = []
    for index, frame in enumerate(frames):
        frame_path = os.path.join(FRAMES_PATH, f"{index}.jpg")

        cv2.imwrite(frame_path, frame)

        frame_paths.append(frame_path)

    logger.info(f"{len(frame_paths)} frame(s) extracted.")

    logger.info("Analyzing video...")
    video_analysis = {index: analyze_image(frame_path) for index, frame_path in enumerate(frame_paths)}

    formatted_video_analysis = {
        key: [format_detection(detection) for detection in detections]
        for key, detections in video_analysis.items()
    }

    logger.info("Video analyzed.")

    logger.info("Editing video...")
    modified_frames = draw_on_frames(frames, formatted_video_analysis)

    video_output_path = os.path.join(OUTPUT_PATH, "video.mp4")
    frames_by_second = get_video_frames_by_second(video_path)

    generate_video(modified_frames, video_output_path, frames_by_second)

    logger.info("Video edited.")

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


def get_video_frames_by_second(video_path):
    video = cv2.VideoCapture(video_path)

    frames_by_second = video.get(cv2.CAP_PROP_FPS)

    video.release()

    return int(frames_by_second)


def draw_on_frames(frames, video_analysis):
    modified_frames = []

    for index, frame in enumerate(frames):
        for detection in video_analysis[index]:
            region = detection["region"]
            x, y, width, height = region["x"], region["y"], region["width"], region["height"]

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

            labels = [f"Age: {detection['age']}", f"Gender: {detection['gender']}", f"Emotion: {detection['emotion']}"]
            cv2.putText(frame, ", ".join(labels), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        modified_frames.append(frame)

    return modified_frames


def generate_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


if __name__ == "__main__":
    main()


import argparse
import os
import numpy as np
from cv2 import cv2
from typing import List

def read_video(path: str) -> List[np.ndarray]:
    frames = []
    capture = cv2.VideoCapture(path)
    while True:
        ret, frame = capture.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames

def write_video(path: str, frames: List[np.ndarray]) -> None:
    if len(frames) <= 0:
        raise ValueError(f'No frames to write!')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(path, fourcc, 24.0, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def find_lanes(frame: np.ndarray) -> np.ndarray:
    copy_frame = np.copy(frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    canny_frame = cv2.Canny(gray_frame, 75, 150)
    corners = np.array([
                           [125, gray_frame.shape[0]],
                           [890, gray_frame.shape[0]],
                           [525, 325],
                           [445, 325]
                       ])
    mask = np.zeros_like(canny_frame)
    cv2.fillPoly(mask, [corners], 255)
    masked_frame = cv2.bitwise_and(canny_frame, mask)
    lanes = cv2.HoughLinesP(masked_frame, 2, np.pi / 180, 20, np.array([]), minLineLength=25, maxLineGap=10)
    largest = -np.inf
    gradient = 0.
    for lane in lanes:
        for x1, y1, x2, y2 in lane:
            current = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if current >= largest:
                largest = current
                gradient = (y2 - y1) / (x2 - x1)
    full_lanes, dashed_lanes = [], []
    for lane in lanes:
        for x1, y1, x2, y2 in lane:
            current_gradient = (y2 - y1) / (x2 - x1)
            if (current_gradient > 0 and gradient > 0) or (current_gradient < 0 and gradient < 0):
                full_lanes.append([x1, y1])
                full_lanes.append([x2, y2])
                cv2.line(copy_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            else:
                dashed_lanes.append([x1, y1])
                dashed_lanes.append([x2, y2])
                cv2.line(copy_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return copy_frame

def find_all_lanes(frames: List[np.ndarray]) -> List[np.ndarray]:
    return list(map(lambda frame: find_lanes(frame), frames))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vp', '--video_path', type=str,
                        default='./data/whiteline.mp4',
                        help='The path to the video.')
    parser.add_argument('-op', '--output_path', type=str,
                        default='./data/outputs/',
                        help='The path where output video is stored.')
    args = parser.parse_args()
    video_path = args.video_path
    output_path = args.output_path

    if not os.path.exists(video_path):
        raise ValueError(f'The path {video_path} does not exist!')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frames = read_video(video_path)
    processed_frames = find_all_lanes(frames)
    
    output_video_path = os.path.join(output_path, 'whiteline_processed.mp4')
    write_video(output_video_path, processed_frames)

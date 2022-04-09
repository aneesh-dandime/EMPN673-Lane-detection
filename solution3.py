import argparse
import os
import numpy as np
from cv2 import cv2
from typing import List, Tuple

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

def filter_frame(frame: np.ndarray) -> np.ndarray:
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    h, _, s = cv2.split(hls_frame)
    _, threshold_frame = cv2.threshold(h, 120, 255, cv2.THRESH_BINARY)
    threshold_blur_frame = cv2.GaussianBlur(threshold_frame, (5, 5), 0)
    sobel_in_x = np.absolute(cv2.Sobel(threshold_blur_frame, cv2.CV_32F, 1, 0, 3))
    sobel_in_y = np.absolute(cv2.Sobel(threshold_blur_frame, cv2.CV_32F, 0, 1, 3))
    sobel_magnitude = np.sqrt(sobel_in_x ** 2 + sobel_in_y ** 2)
    binary_magnitude = np.ones_like(sobel_magnitude)
    binary_magnitude[(sobel_magnitude >= 110) & (sobel_magnitude <= 255)] = 1
    _, threshold_frame2 = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
    _, threshold_frame3 = cv2.threshold(frame[:, :, 2], 120, 255, cv2.THRESH_BINARY)
    binary_threshold_frame23 = cv2.bitwise_and(threshold_frame2, threshold_frame3)
    filtered_frame = cv2.bitwise_or(binary_threshold_frame23, np.uint8(binary_magnitude))
    return filtered_frame

def get_region_of_interest(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [corners], 255)
    region_of_interest = cv2.bitwise_and(frame, mask)
    return region_of_interest

def rectify_view(frame: np.ndarray, corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rectified_corners = np.array([
                                     [321, 2],
                                     [321, 981],
                                     [921, 981],
                                     [921, 1]
                                 ])
    homography = cv2.getPerspectiveTransform(np.float32(corners), np.float32(rectified_corners))
    rectified_frame = cv2.warpPerspective(frame, homography, (frame.shape[1], 1000))
    return rectified_frame, homography

def get_peaks(frame: np.ndarray) -> Tuple:
    counts = np.sum(frame[int(frame.shape[0] // 2):, :], axis=0)
    mid = counts.shape[0] // 2
    peak_left = np.argmax(counts[:mid])
    peak_right = np.argmax(counts[mid:]) + mid
    return counts, peak_left, peak_right

def convert_three_channel(arr: np.ndarray) -> np.ndarray:
    return np.dstack((arr, arr, arr))

def get_centroids(frame: np.ndarray, histogram_peaks: list, num_windows: int) -> Tuple[np.ndarray, np.ndarray]:
    sliding_window_frame = convert_three_channel(np.copy(frame))
    window_width_per_frame = 12
    window_height = frame.shape[0] // num_windows
    window_width = frame.shape[1] // window_width_per_frame
    peak_left, peak_right = histogram_peaks
    mean_left = peak_left
    mean_right = peak_right
    centroids_left = []
    centroids_right = []
    ht = frame.shape[0] - 1 - window_height
    for _ in range(num_windows-1):
        window_left = frame[ht:ht + window_height, mean_left - (window_width//2):mean_left + (window_width//2)]
        if len(np.where(window_left == 255)[1]) > 10:
            mean_left = int(np.mean(np.where(window_left == 255)[1])) + mean_left - (window_width//2)
        centroids_left.append([ht + window_height//2, mean_left])
        window_right = frame[ht:ht + window_height, mean_right - (window_width//2):mean_right + (window_width//2)]
        if len(np.where(window_right == 255)[1]) > 10:
            mean_right = int(np.mean(np.where(window_right == 255)[1])) + mean_right - (window_width//2)
        centroids_right.append([ht + window_height//2, mean_right])
        ht -= window_height
    centroids = np.array([centroids_left,
                              centroids_right])

    return centroids, sliding_window_frame

def detect_lane_lines(frame: np.ndarray, centroids: np.ndarray, sliding_average: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lanes_frame = np.copy(frame)
    fit_left = np.polyfit(centroids[0, :, 0], centroids[0, :, 1], 2)
    fit_right = np.polyfit(centroids[1, :, 0], centroids[1, :, 1], 2)
    fits = [[], []]
    if len(fits[0]) >= sliding_average:
        fits[0].pop(0)
        fits[1].pop(0)
    fits[0].append(fit_left)
    fits[1].append(fit_right)
    fit_left = np.mean(fits[0], axis=0)
    fit_right = np.mean(fits[1], axis=0)
    x = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
    y = fit_left[0] * (x ** 2) + fit_left[1] * x + fit_left[2]
    lane_line_left = np.array([y, x], dtype=int).transpose()
    y = fit_right[0] * (x ** 2) + fit_right[1] * x + fit_right[2]
    lane_line_right = np.array([y, x], dtype=int).transpose()
    cv2.polylines(lanes_frame, [lane_line_left], False, (0, 0, 255), 4)
    cv2.polylines(lanes_frame, [lane_line_right], False, (0, 255, 255), 4)
    return fit_left, lane_line_left, fit_right, lane_line_right, lanes_frame

def project_lanes(frame: np.ndarray, rectified_view: np.ndarray, lane_lines: np.ndarray, homography: np.ndarray) -> np.ndarray:
    projected = convert_three_channel(np.zeros_like(rectified_view))
    cv2.fillPoly(projected, [lane_lines], (0, 0, 255))
    homography_inverse = np.linalg.inv(homography)
    projected = cv2.warpPerspective(projected, homography_inverse, (frame.shape[1], frame.shape[0]))
    projected = cv2.addWeighted(frame, 1, projected, 0.4, 0)
    return projected

def find_lane_curvature(frame: np.ndarray) -> np.ndarray:
    filtered_frame = filter_frame(frame)
    corners = np.array([[573, 471],
                        [222, 681],
                        [1203, 681],
                        [751, 472]])

    rectified_view, homography = rectify_view(filtered_frame, corners)
    _, peak_left, peak_right = get_peaks(rectified_view)
    centroids, sliding_window_frame = get_centroids(rectified_view, [peak_left, peak_right], 20)
    fit_left, lane_line_left, fit_right, lane_line_right, lanes_frame = detect_lane_lines(sliding_window_frame, centroids)
    curvature_left = ((1 + (2 * fit_left[0] * sliding_window_frame.shape[0] * (30 / sliding_window_frame.shape[0]) + fit_left[1]) ** 2) ** 1.5) / np.absolute(2 * fit_left[0])
    curvature_right = ((1 + (2 * fit_right[0] * sliding_window_frame.shape[0] * (30 / sliding_window_frame.shape[0]) + fit_right[1]) ** 2) ** 1.5) / np.absolute(2 * fit_right[0])
    avg_curvature = (curvature_left + curvature_right) / 2
    projected = project_lanes(frame, rectified_view, np.vstack((lane_line_left, np.flipud(lane_line_right))), homography)
    cv2.putText(projected, f'Left Curvature: {round(curvature_left, 2)}', (20,30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(projected, f'Right Curvature: {round(curvature_right, 2)}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(projected, f'Median curvature: {round(avg_curvature, 2)}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    filtered_frame = cv2.resize(filtered_frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    rectified_view = convert_three_channel(rectified_view)
    rectified_view = cv2.resize(rectified_view, None, fx=0.4, fy=0.432, interpolation=cv2.INTER_CUBIC)
    lanes_frame = cv2.resize(lanes_frame, None, fx=0.4, fy=0.432, interpolation=cv2.INTER_CUBIC)

    return projected

def detect_all_lanes(frames: List[np.ndarray]) -> List[np.ndarray]:
    return list(map(lambda f: find_lane_curvature(f), frames))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-vp', '--video_path', type=str,
                        default='./data/challenge.mp4',
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
    processed_frames = detect_all_lanes(frames)
    
    output_video_path = os.path.join(output_path, 'challenge_processed.mp4')
    write_video(output_video_path, processed_frames)

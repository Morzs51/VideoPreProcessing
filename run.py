import cv2
import keyboard
import open3d as o3d
import numpy as np

from reconstruction import Reconstruction
from video_preprocessing import VideoPreprocessing
from reconstruction_config import ReconstructionConfig


def setup_video_preprocessing():
    video_preprocessing = VideoPreprocessing()
    video_preprocessing.set_config_realsense()
    video_preprocessing.set_video_capture()
    video_preprocessing.set_video_writer()
    return video_preprocessing


def transform_frame_for_o3d(frame):
    return o3d.t.geometry.Image(np.asarray(frame))


def setup_reconstruction(video_preprocessing):
    reconstruction_config = ReconstructionConfig()
    video_preprocessing.set_video_capture()
    img = video_preprocessing.get_color_frame()
    depth_frame = video_preprocessing.get_depth_frame(img)
    depth_frame_o3d = transform_frame_for_o3d(depth_frame)
    return Reconstruction(reconstruction_config, depth_frame_o3d)


def run():
    video_preprocessing = setup_video_preprocessing()
    reconstruction = setup_reconstruction(video_preprocessing)
    count = 0
    while True:
        color_frame = video_preprocessing.get_color_frame()
        depth_frame = video_preprocessing.get_depth_frame(color_frame)
        color_frame, _ = video_preprocessing.cut_frame(color_frame)

        if color_frame is not None:
            color_frame = color_frame.astype(np.float32)
            depth_frame = depth_frame.astype(np.float32)
            color_frame = transform_frame_for_o3d(color_frame)
            depth_frame = transform_frame_for_o3d(depth_frame)

            success = reconstruction.launch(color_frame, depth_frame)
            count += 1
        if count == 50:
            break

    # video_preprocessing.video_writer_release()
    reconstruction.visualize_and_save_pcd()


run()

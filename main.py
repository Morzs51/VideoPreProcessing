import cv2
import numpy as np
import time
from config import Config


config = Config()


class video_pre_processing():
    def __init__(self):
        self.frames_for_analysis = None
        self.frame_interval = None
        self.cap = None
        self.device = None

    @staticmethod
    def color_palette(img, color_mode=cv2.COLOR_BAYER_RG2BGR):
        img = cv2.cvtColor(img, color_mode)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        return img

    @staticmethod
    def is_blur(img, blur_threshold=config.blur_threshold):
        """проверка на заблюренность изображения

        blur_treshold можно подобрать более оптимальный

        """
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(grey, cv2.CV_64F).var()
        if var < blur_threshold:
            print('Изображение размыто')
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            return True
        else:
            print('Изображение не размыто')
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            return False

    @staticmethod
    def set_image_size(img, weight, height):
        new_img = cv2.resize(img, dsize=(height, weight), interpolation=cv2.INTER_AREA)
        cv2.imshow("img", new_img)
        cv2.waitKey(0)
        return new_img

    @staticmethod
    def merge_images(self, img1, img2):
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2)
        stitcher = cv2.Stitcher.create()
        status, stitched_image = stitcher.stitch([image1, image2])
        if status == cv2.Stitcher_OK:
            cv2.imshow('st', stitched_image)
            cv2.waitKey(0)
        else:
            print("ошибка склеивания")

    def set_video_capture(self, device=config.device, path=config.path):
        if device:
            cap = cv2.VideoCapture(device)
        else:
            cap = cv2.VideoCapture(path)
        self.cap = cap

    def set_video_capture_size(self, weight, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_video_capture_fps(self, fps=30):
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def set_frame_interval(self, new_time=config.frame_interval):
        self.frame_interval = new_time

    def get_frames_for_analysis(self):
        time_start = time.time()
        self.frames_for_analysis = []
        while True:
            if time.time() - time_start >= self.frame_interval:
                time_start = time.time()
                self.frames_for_analysis.append(self.get_frame())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




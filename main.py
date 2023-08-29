import cv2
import numpy as np
import time
from config import Config

config = Config()


class video_pre_processing():
    """Класс пред обработчик видео"""

    def __init__(self):
        self.frames_for_analysis = None
        self.frame_interval = None
        self.cap = None
        self.device = None

    @staticmethod
    def color_palette(img, color_mode=cv2.COLOR_BAYER_RG2BGR):
        """
        Выбор цветовой палитры
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :param color_mode: передоваемая цветовая палитра cv2
        :return: измененное изображение
        """

        img = cv2.cvtColor(img, color_mode)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        return img

    @staticmethod
    def is_blur(img, blur_threshold=config.blur_threshold):
        """
        Проверка на заблюренность изображения
        blur_threshold можно подобрать более оптимальный
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :param blur_threshold: порог размытости изображения
        :return: возвращает True, если изображение размыто и False, если не размыто
        """

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(grey, cv2.CV_64F).var()
        if var < blur_threshold:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            return True
        else:
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            return False

    @staticmethod
    def set_image_size(img, weight, height):
        """
        Изменение разрешения изображения
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :param weight: ширина нового изображения
        :param height: высота нового изображения
        :return: возвращает полученное изображение
        """

        new_img = cv2.resize(img, dsize=(weight, height), interpolation=cv2.INTER_AREA)
        cv2.imshow("img", new_img)
        cv2.waitKey(0)
        return new_img

    def merge_images(self, image1, image2, image_weight, image_height):
        """
        Склейка вдух изображений
        :param image1: первое передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :param image2: второе передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :param image_weight: ширина склеиного  изображения
        :param image_height: высота склеиного изображения
        :return: полученное изображение
        """

        stitcher = cv2.Stitcher.create()
        status, stitched_image = stitcher.stitch([image1, image2])
        stitched_image = self.set_image_size(stitched_image, image_weight, image_height)
        if status == cv2.Stitcher_OK:
            cv2.imshow('st', stitched_image)
            cv2.waitKey(0)
            return stitched_image
        else:
            print("ошибка склеивания")

    @staticmethod
    def histogram_alignment(self, img):
        """
        Выравнивание гистограммы
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :return: полученное изображение
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        result = np.hstack((img, equ))
        cv2.imshow("res", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return result

    def set_video_capture(self, device=config.device, path=config.path):
        """
        Использование видеопотока или видеофайла
        :param device: номер используемой камеры
        :param path: путь к видеофайлу
        """

        if device:
            cap = cv2.VideoCapture(device)
        else:
            cap = cv2.VideoCapture(path)
        self.cap = cap

    def set_video_capture_size(self, weight, height):
        """
        Задает разрешение видео
        :param weight: ширина видео
        :param height: высота видео
        """

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, weight)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def set_video_capture_fps(self, fps=30):
        """
        Количество кадров у видео
        :param fps: количество кадров в секунду
        """

        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def get_frame(self):
        """
        Получение кадра из видео
        :return: полученный кадр
        """

        ret, frame = self.cap.read()
        return frame

    def set_frame_interval(self, new_time=config.frame_interval):
        """
        Изменение интервала получения кадра
        :param new_time: новое время для получения изображения
        """

        self.frame_interval = new_time

    def get_frames_for_analysis(self):
        """Интервал получения кадра к анализу"""

        time_start = time.time()
        self.frames_for_analysis = []
        while True:
            if time.time() - time_start >= self.frame_interval:
                time_start = time.time()
                self.frames_for_analysis.append(self.get_frame())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

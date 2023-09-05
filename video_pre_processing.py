import cv2
import pyrealsense2 as rs
import numpy as np
import time
from config import Config

config = Config()


class VideoPreProcessing():
    """Класс пред обработчик видео"""

    def __init__(self):
        self.frames_for_analysis = None
        self.frame_interval = None
        self.cap = None
        self.device = None
        self.writer = None
        self.videofile_count = 0
        self.pipeline = rs.pipeline()
        self.config_realsense = rs.config()

    def set_config_realsense(self, stream_type=config.stream_type, weight=config.input_weight,
                             height=config.input_height, stream_format=config.stream_format, fps=config.fps):
        """
        Настройки конфигурации камеры realsense
        :param stream_type: тип запущенного видопотока
        :param weight: ширина видеопотока
        :param height: высота видеопотока
        :param stream_format: формат запущенного видеопотока
        :param fps: количество кадров в секунду у видеопотока
        """
        self.config_realsense.enable_stream()
        self.config_realsense.enable_stream(stream_type[0], weight, height, stream_format[0], fps)
        self.config_realsense.enable_stream(stream_type[1], weight, height, stream_format[1], fps)

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
        return new_img

    def merge_images(self, image1, image2, image_weight=config.merge_iamge_weight, image_height=config.merge_iamge_height):
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
        if image_height is not None or image_weight is not None:
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

    def set_video_capture(self, device=config.device, path=config.input_path):
        """
        Использование видеопотока или видеофайла
        :param device: номер используемой камеры
        :param path: путь к видеофайлу
        """

        if path is None:
            self.set_config_realsense()
            self.pipeline.start(self.config_realsense)
        else:
            self.config_realsense.enable_device_from_file(path)
            self.pipeline.start(self.config_realsense)

    def set_video_writer(self, path=config.output_path, fps=config.output_fps, weight=config.output_weight,
                         height=config.output_height, codec=config.codec):
        """
        Инициализация потока для записи видео
        :param path: путь куда записывать файл и в каком формате
        :param fps: количество кадорв в секунду у записываемого видео
        :param weight: ширина кадра у записываемого видео
        :param height: высота кадра у записываемого видео
        :param codec: кодек у записываемого видео
        """

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(path, fourcc, fps, (weight, height))

    def write_video(self, img):
        """
        Записывание изображения в видео
        :param img: передаваемое изображение
        """

        if not self.writer:
            self.set_video_writer()
        self.writer.write(img)

    def video_writer_release(self):
        """
        Окончание записи видео
        """

        self.videofile_count += 1
        filename, file_extension = config.output_path.split('.')
        config.output_path = f"{filename}_{self.videofile_count}.{file_extension}"
        self.writer.release()

    def set_video_capture_size(self, weight=config.input_weight, height=config.input_height):
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
        :return: полученны два кадра (цвета и глубины)
        """

        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()
        return color_frame, depth_frame

    def set_frame_interval(self, new_time=config.frame_interval):
        """
        Изменение интервала получения кадра
        :param new_time: новое время для получения изображения
        """

        self.frame_interval = new_time

    def get_frames_for_analysis(self):
        """Интервал получения кадра к анализу"""

        time_start = time.time()
        self.frames_for_analysis = self.frames_for_analysis[:2]  # указать размерность массива
        while True:
            if time.time() - time_start >= self.frame_interval:
                time_start = time.time()
                self.frames_for_analysis.append(self.get_frame())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

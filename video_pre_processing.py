import cv2
import cv2 as cv
import pyrealsense2 as rs
import numpy as np
import time
from config import Config

config = Config()


class VideoPreProcessing:
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
        self.is_the_function_initialized_set_config_realsense = False

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
        self.is_the_function_initialized_set_config_realsense = True

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

    def merge_images(self, image1, image2, image_weight=config.merge_iamge_weight,
                     image_height=config.merge_iamge_height):
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
        if self.is_the_function_initialized_set_config_realsense:
            frame = self.pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()
            return color_frame, depth_frame
        else:
            if config.StereoSGBM_or_StereoBM == 0:
                return self.StereoSGBM_func()
            else:
                return self.StereoBM_func()

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

    def oneShot(self):
        # StereoSGBM Parameter explanations:
        # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -128
        max_disp = 128
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 5
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0
        img1_undistorted = cv.imread("left400.png", cv.IMREAD_GRAYSCALE)
        img2_undistorted = cv.imread("right400.png", cv.IMREAD_GRAYSCALE)
        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(img1_undistorted, img2_undistorted)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        cv.imshow("Disparity", disparity_SGBM)
        cv.imwrite("disparity_SGBM_norm.png", disparity_SGBM)
        print("finish")
        i = 0
        while (10 > i):
            i -= 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def StereoSGBM_func(self, skip_frame=config.skip_frame):
        """
        Функция получения стерео SGBM кадра
        :param skip_frame: текущий кадр для чтения по переданному костылю минимальноек значение равняется 200
        :return: цветной левый кадр и стерео SGBM кадр
        """
        if skip_frame < 200:
            skip_frame = 200
            config.skip_frame = 200
        video = cv.VideoCapture("video.avi")
        video.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        config.skip_frame = +1
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 3
        min_disp = -176
        max_disp = 176
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 15
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 16
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 2

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        # инициализации класса Стерео описание параметров с оф документации
        ret, frame = video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        #print("H:" + str(height) + "  W:" + str(width))
        h = int(height)
        w = int(width / 2)  # делем на пополам потому что изображение сдвоенное
        x1 = w
        left_img = gray[y:y + h, x:x + w]  # получаем левое изображение из сдвоенного
        right_img = gray[y:y + h, x1:x1 + w]  # получаем правое изображение из сдвоенного
        disparity_SGBM = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                          beta=0, norm_type=cv.NORM_MINMAX)  # увеличиваем контраст
        disparity_SGBM = np.uint8(disparity_SGBM)
        res = left_img + disparity_SGBM  # комплексированное изображение
        #cv.imshow('left', res)
        video.release()
        cv.destroyAllWindows()
        return left_img, res

    def StereoBM_func(self, skip_frame=config.skip_frame):
        """
        Функция получения стерео BM кадра
        :param skip_frame: текущий кадр для чтения по переданному костылю минимальноек значение равняется 200
        :return: цветной левый кадр и стерео BM кадр
        """
        if skip_frame < 200:
            skip_frame = 200
            config.skip_frame = 200
        video = cv.VideoCapture("video.avi")
        video.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)
        config.skip_frame = +1
        # Вариант исполнения получения кадра глубины
        h = 0
        w = 0
        # Ширина и высота кадра
        x = 0
        x1 = 0
        y = 0
        # Отступы для первого кадра x y и для второго x1 y

        stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)

        ret, frame = video.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        #print("H:" + str(height) + "  W:" + str(width))
        h = int(height)
        w = int(width / 2)
        x1 = w
        left_img = gray[y:y + h, x:x + w]
        right_img = gray[y:y + h, x1:x1 + w]
        disparity = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity = cv.normalize(disparity, disparity, alpha=125,
                                    beta=0, norm_type=cv.NORM_MINMAX)
        disparity = np.uint8(disparity)
        res = disparity
        #cv.imshow('left', res)
        video.release()
        cv.destroyAllWindows()
        return left_img, res


    def crop(self, img):
        """
        Обрезка изображений пополам
        :param img: передаваемое изображение прочитанное библиотекой cv2 (cv.imread)
        :return: левая половина изначального изображения и правая половина 
        """
        height, width, _ = img.shape
        half_width = width // 2
        left_image = img[:, :half_width]
        right_image = img[:, half_width:]
        return left_image, right_image

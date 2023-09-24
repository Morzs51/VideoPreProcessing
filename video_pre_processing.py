import cv2
import os
import numpy as np
import pyrealsense2 as rs
from video_preprocessing_config import Config
from logger import Logger

log = Logger()
class VideoPreprocessing:

    def __init__(self):
        self.cap = None
        self.writer = None
        self.config = Config()
        self.pipeline = rs.pipeline()
        log.init("успешная инициализация класса VideoPreprocessing")

    def set_config_realsense(self):
        depth_format, color_format = self.config.stream_format
        rs.config.enable_stream(rs.stream.depth, self.config.input_width, self.config.input_height,
                                depth_format, self.config.fps)
        rs.config.enable_stream(rs.stream.color, self.config.input_width, self.config.input_height,
                                color_format, self.config.fps)
        log.init("успешная инициализация параметров realsense")

    def change_image_color_mode(self, img, color_mode=None):
        try:
            color_mode = color_mode or self.config.color_mode
            log.info("успеншное изменение цветовой палитры изображения")
            return cv2.cvtColor(img, color_mode)
        except Exception as e:
            log.error(f"ошибка изменения цветовой палитры изображения \n{e}")

    def is_image_blured(self, img):
        gray_img = self.change_image_color_mode(img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray_img, cv2.CV_64F).var() < self.config.blur_threshold:
            log.info("изображение размыто")
            return True
        log.info("изображение не размыто")
        return False

    def set_image_size(self, img, shape=None):
        return cv2.resize(
            img,
            dsize=shape or (self.config.input_width, self.config.input_height),
            interpolation=self.config.interpolation_method
        )

    def merge_images(self, img1, img2):
        try:
            stitcher = cv2.Stitcher.create()
            status, merged_image = stitcher.stitch([img1, img2])
            merged_image = self.set_image_size(merged_image, (self.config.merge_image_width,
                                                              self.config.merge_image_height))
            if status == cv2.Stitcher_OK:
                return merged_image
            log.info("успешное усклеивание изображений")
            return None
        except Exception as e:
            log.error(f"ошибка склеивания изображений\n{e}")

    def image_histogram_alignment(self, img):
        contrast_img = cv2.equalizeHist(self.change_image_color_mode(img, cv2.COLOR_BGR2GRAY))
        contrast_img = self.change_image_color_mode(contrast_img, cv2.COLOR_GRAY2BGR)
        #contrast_img = np.hstack((img, contrast_img))
        log.info("успешное изменение контрастности изображения")
        return contrast_img

    def set_video_capture(self):
        try:
            if isinstance(self.config.video_source, str):
                self.cap = cv2.VideoCapture(self.config.video_source)
            else:
                if self.config.use_realsense:
                    self.set_config_realsense()
                    self.pipeline.start(rs.config())
                else:
                    self.cap = cv2.VideoCapture(self.config.video_source)
            log.info("успешная инициализация видеозахвата")
        except Exception as e:
            log.error(f"ошибка инициализации видеозахвата\n{e}")

    def set_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        try:
            self.writer = cv2.VideoWriter(
                self.config.output_path,
                fourcc,
                self.config.fps,
                (self.config.output_width, self.config.output_height)
            )
            log.info("успешное создание cv2.VideoWriter")
        except Exception as e:
            log.error(f"ошибка создания cv2.VideoWriter\n{e}")

    def write_video(self, img):
        try:
            if not self.writer:
                self.set_video_writer()
            self.writer.write(img)
        except Exception as e:
            log.error(f"ошибка записи видео{e}")

    def video_writer_release(self):
        self.videofile_count += 1
        filename, file_extension = self.config.output_path.split('.')
        self.config.output_path = f"{filename}_{self.videofile_count}.{file_extension}"
        self.writer.release()
        log.info("cv2.VideoWriter завершил работу")

    #нужон тест
    def set_video_capture_size(self, width=None, height=None):
        width = width or self.config.input_width
        height = height or self.config.input_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        log.info("успешное изменение разрешения видеозахвата")

    def set_video_capture_fps(self):
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        log.info("успешное изменение видеокадров в секунду видеозахвата")

    def get_depth_frame(self, img=None):
        try:
            if self.config.use_realsense:
                frame = self.pipeline.wait_for_frames()
                depth_frame = frame.get_depth_frame()
                log.info("успешное получение кадра глубины")
                return depth_frame
            else:
                left_img, right_img = self.cut_frame(img)
                if self.config.StereoSGBM_or_StereoBM == 0:
                    return self.StereoSGBM(left_img, right_img)
                else:
                    return self.StereoBM(left_img, right_img)
        except Exception as e:
            log.error(f"{e}")

    def get_color_frame(self):
        if self.config.use_realsense:
            color_frame = self.pipeline.wait_for_frames()
            color_frame = color_frame.get_color_frame()
            log.info("успешное получение цветного изображения")
            return np.asanyarray(color_frame.get_data())

        else:
            ret, color_frame = self.cap.read()
            if not ret:
                log.error(f"ошибка получения цветного изображения")
                return None
            log.info("успешное получение цветного изображения")
            return color_frame

    def cut_frame(self, img):
        width = img.shape[1]
        half_width = width // 2
        left_image = img[:, :half_width]
        right_image = img[:, half_width:]
        log.info("успешное разделение изображения")
        return left_image, right_image

    def StereoSGBM(self, left_img, right_img):
        # Вариант исполнения получения кадра глубины

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

        stereo = cv2.StereoSGBM_create(
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
        left_img = self.change_image_color_mode(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.change_image_color_mode(right_img, cv2.COLOR_BGR2GRAY)
        disparity_SGBM = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=125,
                                       beta=0, norm_type=cv2.NORM_MINMAX)  # увеличиваем контраст
        disparity_SGBM = np.uint8(disparity_SGBM)
        log.info("успешное получение кадра глубины")
        return left_img + disparity_SGBM  # комплексированное изображение

    def StereoBM(self, left_img, right_img):
        # Вариант исполнения получения кадра глубины
        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

        left_img = self.change_image_color_mode(left_img, cv2.COLOR_BGR2GRAY)
        right_img = self.change_image_color_mode(right_img, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(left_img, right_img)
        # Normalize the values to a range from 0..255 for a grayscale image
        disparity = cv2.normalize(disparity, disparity, alpha=125,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
        log.info("успешное получение кадра глубины")
        return np.uint8(disparity)

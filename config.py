import pyrealsense2 as rs


class Config:
    def __init__(self):
        self.device = 0                                             # Номер используемой камеры
        self.input_path = ''                                        # Путь до читаемого видеофайла
        self.output_path = ''                                       # Путь до записываемого видеофайла
        self.fps = 30                                               # Количество кадров в секунду при чтении в видео
        self.output_fps = 25                                        # Количество кадров в секунду для записи видео
        self.blur_threshold = 120                                   # Порог размытости изображения
        self.frame_interval = 5                                     # Интервал времени в который берется кадр 5 = 0.005
        self.input_weight = 640                                     # Входная ширина видео
        self.input_height = 480                                     # Входная высота видео
        self.output_weight = 640                                    # Ширина записываемого видеофайла
        self.output_height = 480                                    # Высота записываемого видеофайла
        self.codec = 'bag'                                          # Кодек записываемого видеофайла
        self.stream_type = (rs.stream.depth, rs.stream.color)       # Тип запущенного видопотока
        self.stream_format = (rs.format.z16, rs.format.bgr8)        # Формат запущенного видеопотока
        self.merge_iamge_weight = 1500                              # Ширина слекинного изображения
        self.merge_iamge_height = 400                               # Высота слекинного изображения

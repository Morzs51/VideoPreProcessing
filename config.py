class Config:
    def __init__(self):
        self.device = 0                 # Номер используемой камеры
        self.path = ''                  # Путь до видеофайла
        self.fps = 30                   # Количество кадров в видео
        self.blur_threshold = 120       # Порог размытости изображения
        self.frame_interval = 5         # Интервал времени в который берется кадр 5 = 0.005

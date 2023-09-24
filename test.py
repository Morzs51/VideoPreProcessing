from video_preprocessing import VideoPreprocessing
import cv2

def run_test():
    test = VideoPreprocessing()

    img1 = cv2.imread("1.jpg")
    img2 = cv2.imread("2.jpg")
    img3 = cv2.imread("3.jpg")

    test.change_image_color_mode(img1)
    cv2.imshow('после изменений', img1)
    cv2.waitKey(0)

    test.is_image_blured(img1)
    test.is_image_blured(img3)

    test.set_image_size(img2)
    cv2.imshow("изображение с измененныым размером", img2)

    merge_img = test.merge_images(cv2.imread("1.jpg"), cv2.imread("2.jpg"))
    cv2.imshow("склеенное изображение", merge_img)

    test.image_histogram_alignment(img3)
    cv2.imshow("изображение с измененной контрастностью", img3)

    test.set_video_capture()

    frame = test.get_color_frame()
    cv2.imshow("кадр глубины", test.get_depth_frame(test.cut_frame(frame)))

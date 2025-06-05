import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer
import argparse

# Веса для распознования лиц
faceProto = "opencv_face_detector.pbtxt"
#Конфигурация нейросети - слои и связи нейронов
faceModel = "opencv_face_detector_uint8.pb"
# Новые модели
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

# Настраиваем свет
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Итоговые результаты работы нейросетей для пола и возраста
genderList = ['Male ', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
color = (0, 255, 0)

# Запуск нейросети по распознаванию лиц
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Запуск нейросети по определению пола и возраста
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Функция определения лиц
def highlightFace(net, frame, conf_threshold = 0.7):
    # копия кадра
    frameOpencvDnn = frame.copy()
    # Высота и ширина кадра
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    # Преобразование в двоичный пиксельный объект
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    # Объект входной параметр для нейросети
    net.setInput(blob)
    # Прямой проход для распознования лиц
    detections = net.forward()
    # Переменная для рамок вокруг лица
    faceBoxes = []
    # Перебираем блоки после распознования
    for i in range(detections.shape[2]):
        # Результат
        confidence = detections[0, 0, i, 2]
        # Если результат превышает порог срабатываний - это лицо
        if confidence > conf_threshold:
            # Формирование координат рамки
            x1 = int(detections[0, 0, i, 3]*frameWidth)
            y1 = int(detections[0, 0, i, 4]*frameHeight)
            x2 = int(detections[0, 0, i, 5]*frameWidth)
            y2 = int(detections[0, 0, i, 6]*frameHeight)
            # Общая переменная
            faceBoxes.append([x1, y1, x2, y2])
            # Рамка на кадре
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), color, 2, 8)
    # Кадр с рамками
    return frameOpencvDnn, faceBoxes

# Основной GUI-класс
class AgeGenderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Age and Gender Detection")
        self.image_label = QLabel("Camera stream or selected image will appear here")
        self.image_label.setFixedSize(640, 480)

        self.camera_button = QPushButton("Start Camera")
        self.load_button = QPushButton("Load Image")

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.camera_button)
        layout.addWidget(self.load_button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.capture = None

        self.camera_button.clicked.connect(self.start_camera)
        self.load_button.clicked.connect(self.load_image)
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            processed_frame = self.process_frame(frame)
            self.display_image(processed_frame)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            frame = cv2.imread(file_name)
            processed_frame = self.process_frame(frame)
            self.display_image(processed_frame)

    def process_frame(self, frame):
        # Распознование лица в кадре
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        # Перебираем все найденный лица в кадре
        for faceBox in faceBoxes:
            # Получение изображение лица на основе рамки
            face = frame[max(0, faceBox[1]):min(faceBox[3],frame.shape[0]-1), 
                        max(0, faceBox[0]):min(faceBox[2], frame.shape[1]-1)]
            # Получаем на этой основе новый бинарный пиксельный объект
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Отправляем его в нейросеть для определения пола
            genderNet.setInput(blob)
            # Выбираем пол на основе этого результата
            gender = genderList[genderNet.forward()[0].argmax()]

            # Возраст
            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"
            # Текст возле каждой рамки в кадре
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            
        return resultImg

    def display_image(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

# Запуск приложения
if __name__ == "__main__":
    app = QApplication([])
    window = AgeGenderApp()
    window.show()
    app.exec()
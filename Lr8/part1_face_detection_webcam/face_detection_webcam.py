import cv2

# Функция определения лиц
def highlightFace(net, frame, conf_threshold = 0.7):
    # копия кадра
    frameOpencvDnn = frame.copy()
    # Высота и ширина кадра
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    # Кадр с рамками
    return frameOpencvDnn, faceBoxes

# Веса для распознования лиц
faceProto = "opencv_face_detector.pbtxt"
# Конфигурация нейросети - слои и связи нейронов
faceModel = "opencv_face_detector_uint8.pb"

# Запуск
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Видео с камеры
video = cv2.VideoCapture(0)
# Пока не нажата любая клавиша, выполняется цикл
while cv2.waitKey(1)<0:
    # Кадр с камеры
    hasFrame, frame = video.read()
    # Кадра нет
    if not hasFrame:
        # Останавливаем и выходим из цикла
        cv2.waitKey()
        break
    # Распознование лица в кадре
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    # Если лиц нет
    if not faceBoxes:
        # Лицо не найдено
        print("Face not recognized")
    # Картинка с камеры
    cv2.imshow("Face detection", resultImg)
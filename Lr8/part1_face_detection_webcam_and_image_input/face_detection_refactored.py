import cv2
import sys

# Функция определения лиц
def highlightFace(net, frame, conf_threshold = 0.7):
    # копия кадра
    frameOpencvDnn = frame.copy()
    # Высота и ширина кадра
    frameHeight, frameWidth = frame.shape[:2]
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

# Функция для обработки изображения из файла
def process_image(image_path, net):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Если изображение не удалось загрузить
    if image is None:
        print(f"Error: Failed to load image at path '{image_path}'")
        return
    # Определение лиц на изображении с помощью функции
    resultImg, faceBoxes = highlightFace(net, image)
    if not faceBoxes:
        print("Face not recognized")
    else:
        print(f"Person found:{len(faceBoxes)}")
    # Отображение изображения с рамками
    cv2.imshow("Face detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Функция для обработки видеопотока с камеры
def process_camera(net):
    # Видео с камеры
    video = cv2.VideoCapture(0)
    # Проверка, открылось ли устройство
    if not video.isOpened():
        print("Error: Failed to open video stream from camera")
        return
    # Пока не нажата любая клавиша, выполняется цикл
    while cv2.waitKey(1)<0:
        # Кадр с камеры
        hasFrame, frame = video.read()
        # Кадра нет
        if not hasFrame:
            print("Error: Failed to get frame from camera")
            break
        # Распознование лица в кадре
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        # Если лиц нет
        if not faceBoxes:
            # Лицо не найдено
            print("Face not recognized")
        else: 
            print(f"Person found: {len(faceBoxes)}")

        # Картинка с камеры
        cv2.imshow("Face detection", resultImg)

        # Условие выхода
        if cv2.waitKey(1) == 27:
            print("Exit by prossing ESC")
            break
    # Освобождение камеры
    video.release()
    # Закрытие всех окон
    cv2.destroyAllWindows()

# Оносновной блок
if __name__ == "__main__":
    # Веса для распознования лиц
    faceProto = "opencv_face_detector.pbtxt"
    #Конфигурация нейросети - слои и связи нейронов
    faceModel = "opencv_face_detector_uint8.pb"
    # Запуск
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    if len(sys.argv) > 1:
        # Путь к изображению
        image_path = sys.argv[1]
        # Обработка изображения
        process_image(image_path, faceNet)
    else:
        # Запуск камеры
        process_camera(faceNet)
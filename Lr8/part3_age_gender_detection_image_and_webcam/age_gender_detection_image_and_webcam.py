import cv2
import argparse

# Подлючение парсера аргументов командной строки
parser = argparse.ArgumentParser()
# Добавляем аргумент для работы с изображениями
parser.add_argument('--image')
# Сохраняем аргументы в отдельную переменную
args = parser.parse_args()
# Цвет по умолчанию
color = (0, 255, 0)

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
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), color, int(round(frameHeight/150)), 8)
    # Кадр с рамками
    return frameOpencvDnn, faceBoxes

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

# Запуск нейросети по распознаванию лиц
faceNet = cv2.dnn.readNet(faceModel, faceProto)
# Запуск нейросети по определению пола и возраста
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Если указан аргумент с картинкой - берем картинку как источник
video = cv2.VideoCapture(args.image if args.image else 0)
# Пока не нажата любая клавиша, выполняется цикл
while cv2.waitKey(1)<0:
    # Кадр с камеры
    hasFrame, frame = video.read()
    # Кадра нет
    if not hasFrame:
        cv2.waitKey()
        break
    # Распознование лица в кадре
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    # Перебираем все найденный лица в кадре
    for faceBox in faceBoxes:
        # Получение изображение лица на основе рамки
        face = frame[max(0, faceBox[1]):
                     min(faceBox[3],frame.shape[0]-1), max(0,faceBox[0])
                     :min(faceBox[2], frame.shape[1]-1)]
        # Получаем на этой основе новый бинарный пиксельный объект
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # Отправляем его в нейросеть для определения пола
        genderNet.setInput(blob)
        # Результат
        genderPreds = genderNet.forward()
        # Выбираем пол на основе этого результата
        gender = genderList[genderPreds[0].argmax()]
        # Отправляем результат в переменную с полом
        print(f'Gender: {gender}')

        # Возраст
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Текст возле каждой рамки в кадре
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        # Вывод картинки
    cv2.imshow("Detecting age and gender", resultImg)
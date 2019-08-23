import datetime
import time
import cv2
import os
import face_recognition
from PIL import Image
from imageai.Detection import ObjectDetection

camera__addr = "rtsp://admin:admin@192.168.0.64:554/cam/realmonitor?channel=1&subtype=0"


def get__date():
    date = datetime.datetime.now()
    return '{0}.{1}.{2}'.format(date.day if date.day >= 10 else '0{0}'.format(date.day),
                                    date.month if date.month >= 10 else '0{0}'.format(date.month),
                                    date.year)


def get__time():
    date = datetime.datetime.now()
    return '{0}:{1}:{2}'.format(date.hour if date.hour >= 10 else '0{0}'.format(date.hour),
                                date.minute if date.minute >= 10 else '0{0}'.format(date.minute),
                                date.second if date.second >= 10 else '0{0}'.format(date.second))


def save_file(path, filename, bytes):
    if not os.path.exists(path):
        os.makedirs(path)
    fi = open(os.path.join(path, filename), 'wb')
    fi.write(bytes)
    fi.close()

def read_file(path, type='rb'):
    fo = open(path, type)
    content = fo.read()
    fo.close()
    return content


def make__path(date, dir):
    exec_path = os.getcwd()
    if not os.path.exists(os.path.join(exec_path, dir, date)):
        os.makedirs(os.path.join(exec_path, dir, date))
    return os.path.join(exec_path, dir, date)


def make_snapshot(file_name):
    path = make__path(get__date(), 'snapshots')
    cap = cv2.VideoCapture(file_name)
    ret, frame = cap.read()
    img = False
    if not ret:
        pass
    else:
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(path, '{0}.jpg'.format(get__time())), frame)
        img = os.path.join(path, '{0}.jpg'.format(get__time()))
    cap.release()
    cv2.destroyAllWindows()
    return img


def body_detection(file_name):
    try:
        path = make__path(os.path.join(get__date(), get__time()), 'bodies')
        exec_path = os.getcwd()
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(exec_path, 'models', 'resnet50_coco_best_v2.0.1.h5'))
        detector.loadModel()
        custom_obj = detector.CustomObjects(person=True)

        list = detector.detectCustomObjectsFromImage(
            custom_objects=custom_obj,
            input_image=file_name,
            output_image_path=os.path.join(exec_path, 'bodies', 'body.jpg')
        )
        image = face_recognition.load_image_file(os.path.join(exec_path, 'bodies', 'body.jpg'))
        counter = 1
        files = []
        for body in list:
            box_points = body['box_points']
            top = box_points[1]
            bottom = box_points[3]
            left = box_points[0]
            right = box_points[2]

            body_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(body_image)
            pil_image.save(os.path.join(path, 'body{0}.jpg'.format(counter)))
            files.append(os.path.join(path, 'body{0}.jpg'.format(counter)))
            counter += 1
        return files
    except:
        return False

def face_detection(image__list):
    path = make__path(os.path.join(get__date(), get__time()), 'faces')

    if not isinstance(image__list, list):
        image__list = [image__list]

    files = []
    for _image in image__list:
        image = face_recognition.load_image_file(_image)
        face_locations = face_recognition.face_locations(image)

        counter = 1
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            while os.path.isfile(os.path.join(path, 'face{0}.jpg'.format(counter))):
                counter += 1

            pil_image.save(os.path.join(path, 'face{0}.jpg'.format(counter)))
            files.append(os.path.join(path, 'face{0}.jpg'.format(counter)))
            counter += 1
    return files


def face_to_repository(filename, person_name):
    path = make__path(person_name, 'faces_repository')
    image = face_recognition.load_image_file(filename)
    face_locations = face_recognition.face_locations(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(os.path.join(path, 'face.jpg'))


def face_compare(known_path, unknown_path):
    try:
        known_image = face_recognition.load_image_file(known_path)
        unknown_image = face_recognition.load_image_file(unknown_path)
        biden_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        is_compare = face_recognition.compare_faces([biden_encoding], unknown_encoding)[0]
        if is_compare:
            return unknown_path
        else:
            return False
    except:
        return False


def faces_compare(unknown_pathes):
    exec_path = os.getcwd()
    repo_path = os.path.join(exec_path, 'faces_repository')
    known_pathes = [_dir for _dir in os.listdir(repo_path) if os.path.isdir(os.path.join(repo_path, _dir))]

    compares = []
    for known_path in known_pathes:
        known_path_photo = os.path.join(repo_path, known_path, 'face.jpg')
        for unknown_path in unknown_pathes:
            _c = face_compare(known_path_photo, unknown_path)
            if _c is not False:
                compares.append({'known': known_path_photo, 'unknown': unknown_path, 'person': known_path})

    return compares


def save_compares(compares, start_photo=False, ):
    path = make__path(os.path.join(get__date(), get__time()), 'compares')
    if start_photo:
        photo = read_file(os.path.join(os.getcwd(), 'bodies', 'body.jpg'))
        save_file(path, 'original_photo.jpg', photo)
    for compare in compares:
        _c = 1
        photo_bytes = read_file(compare['unknown'])
        while os.path.isfile(os.path.join(path, compare['person'], 'photo-{0}.jpg'.format(_c))):
            _c += 1
        save_file(os.path.join(path, compare['person']), 'photo-{0}.jpg'.format(_c), photo_bytes)

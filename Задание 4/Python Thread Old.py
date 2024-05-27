import time
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = '0' # https://github.com/opencv/opencv/issues/17687
import cv2
import numpy as np
import argparse

import threading
import queue


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    '''Sensor X'''

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
    
    def get(self):
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, cam_name, cam_res, cam_fps):
        self._name = cam_name
        self._width = int(cam_res.split('x')[0])
        self._height = int(cam_res.split('x')[0])
        self._fps = int(cam_fps)

        # https://stackoverflow.com/questions/46674503/opencv-webcam-reading-official-code-is-very-slow
        self._VC = cv2.VideoCapture(self._name)
        self._VC.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._VC.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._VC.set(cv2.CAP_PROP_FPS, self._fps)
    
    def get(self):
        return self._VC.read()
    
    
    def __del__(self):
        self._VC.release()

class WindowImage():
    def __init__(self, window_freq):
        self._freq = window_freq
    
    def show(self, sensor0_data, sensor1_data, sensor2_data, sensorCam_frame, fps):
        s0 = str(sensor0_data) if sensor0_data else '-1'
        s1 = str(sensor1_data) if sensor1_data else '-1'
        s2 = str(sensor2_data) if sensor2_data else '-1'

        frame = (sensorCam_frame[1] if (sensorCam_frame and sensorCam_frame[0])
            else np.full((480, 640, 3), (int(s2) % 256, int(s1) % 256, int(s0) % 256), np.uint8)
        )
        cv2.putText(frame, f's0: {s0} | s1: {s1} | s2: {s2} | FPS: {fps}',
            (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
        )
        cv2.imshow('task_4', frame)
    
    def __del__(self):
        cv2.destroyAllWindows()


def worker(sensor, queue): # Управление этой фукнцией мы передаём потокам ниже
    while True:
        data = sensor.get()

        if queue.empty():
            queue.put(data)

if __name__ == '__main__':
    # Парсим и инициализируем аргументы
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='/dev/video0')
    parser.add_argument('--res', type=str, default='640x480')
    parser.add_argument('--freq', type=int, default=60)
    args = parser.parse_args()

    cam_name = args.name
    cam_res = args.res
    cam_fps = (1 / args.freq)

    # Инициализируем сенсоры, камеру, окно
    sensor0 = SensorX(0.01) # 100 Hz
    sensor1 = SensorX(0.1)  # 10 Hz
    sensor2 = SensorX(1)    # 1 Hz
    sensorCam = SensorCam(cam_name, cam_res, cam_fps)
    window = WindowImage(cam_fps)

    # Создаём потоки для сенсоров выше
    thread0_queue = queue.Queue()
    thread0 = threading.Thread(target=worker, args=(sensor0, thread0_queue))
    thread0.start()
    thread1_queue = queue.Queue()
    thread1 = threading.Thread(target=worker, args=(sensor1, thread1_queue))
    thread1.start()
    thread2_queue = queue.Queue()
    thread2 = threading.Thread(target=worker, args=(sensor2, thread2_queue))
    thread2.start()
    threadCam_queue = queue.Queue()
    threadCam = threading.Thread(target=worker, args=(sensorCam, threadCam_queue))
    threadCam.start()

    # Ожидаем данные с камеры и сенсоров и отображаем их в окне
    sensor0_data = sensor1_data = sensor2_data = sensorCam_frame = None

    timestamp = time.time() # Для замера реального FPS
    
    while True:
        if not thread0_queue.empty():
            sensor0_data = thread0_queue.get_nowait()
        if not thread1_queue.empty():
            sensor1_data = thread1_queue.get_nowait()
        if not thread2_queue.empty():
            sensor2_data = thread2_queue.get_nowait()
        #if not threadCam_queue.empty():
        #    sensorCam_frame = threadCam_queue.get_nowait()
        sensorCam_frame = threadCam_queue.get()

        timediff = time.time() - timestamp # Для замера реального FPS
        window.show(sensor0_data, sensor1_data, sensor2_data, sensorCam_frame, (1 / timediff))
        timestamp = time.time() # Для замера реального FPS
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread0.join()
            thread1.join()
            thread2.join()
            threadCam.join()
            window.__del__()
            break

        time.sleep(cam_fps)

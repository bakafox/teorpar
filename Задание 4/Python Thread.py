import time
import cv2
import numpy as np
import argparse
import logging

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
    def __init__(self, cam_name, cam_res):
        self._name = cam_name
        self._width = int(cam_res.split('x')[0])
        self._height = int(cam_res.split('x')[1])

        self._VC = cv2.VideoCapture(self._name)
        if not self._VC.isOpened():
            logging.error("Unable to connect to the specified camera.")

        self._VC.set(3, self._width)
        self._VC.set(4, self._height)
    
    def get(self):
        read = self._VC.read()
        if not read[0]:
            logging.error("Unable to get a new frame from the camera.")

        return read
    
    def close(self):
        self._VC.release()

class WindowImage():
    def __init__(self, window_freq):
        self._freq = window_freq
    
    def show(self, sensor0_data, sensor1_data, sensor2_data, sensorCam_frame):
        s0 = str(sensor0_data) if sensor0_data else '-1'
        s1 = str(sensor1_data) if sensor1_data else '-1'
        s2 = str(sensor2_data) if sensor2_data else '-1'

        # Использование ТОЛЬКО заглушки ниже вместо реальных кадров поднимает
        # FPS практически до любого заданного в разумных пределах числа!
        frame = (sensorCam_frame[1] if (sensorCam_frame and sensorCam_frame[0])
            else np.full((480, 640, 3), (int(s2) % 256, int(s1) % 256, int(s0) % 256), np.uint8)
        )
        cv2.rectangle(
            frame, (8, 8), (150 + (len(s0+s1+s2)*10), 20), (0, 0, 0), -1
        )
        cv2.putText(
            frame, f's0: {s0} | s1: {s1} | s2: {s2}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)
        )
        cv2.imshow('task_4', frame)
    
    def close(self):
        cv2.destroyAllWindows()


# Управление этой фукнцией мы передаём потокам ниже
def worker(sensor, queue, event_stop: threading.Event):
    while not event_stop.is_set():
        data = sensor.get()
        # Помещаем данные в очередь, только если все
        # прошлые данные уже обработаны (= очередь пуста).
        if queue.empty():
            queue.put(data)

if __name__ == '__main__':
    try:
        logging.basicConfig(filename=f'log/{time.strftime("%Y%m%d-%H%M%S")}.log', level=logging.INFO)
    except:
        logging.warn("'log' folder does not exists; the log file will not be created.")
        pass

    # Парсим и инициализируем аргументы
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='/dev/video0')
    parser.add_argument('--res', type=str, default='640x480')
    parser.add_argument('--freq', type=int, default=30)
    args = parser.parse_args()

    cam_name = args.name
    cam_res = args.res
    cam_fps = (1 / args.freq)

    # Инициализируем сенсоры, камеру, окно
    sensor0 = SensorX(0.01) # 100 Hz
    sensor1 = SensorX(0.1)  # 10 Hz
    sensor2 = SensorX(1)    # 1 Hz
    sensorCam = SensorCam(cam_name, cam_res)
    window = WindowImage(cam_fps)

    # Создаём потоки и нужные к ним прибамбасы
    event_stop = threading.Event()

    thread0_queue = queue.Queue()
    thread0 = threading.Thread(target=worker, args=(sensor0, thread0_queue, event_stop))
    thread0.start()
    thread1_queue = queue.Queue()
    thread1 = threading.Thread(target=worker, args=(sensor1, thread1_queue, event_stop))
    thread1.start()
    thread2_queue = queue.Queue()
    thread2 = threading.Thread(target=worker, args=(sensor2, thread2_queue, event_stop))
    thread2.start()
    threadCam_queue = queue.Queue()
    threadCam = threading.Thread(target=worker, args=(sensorCam, threadCam_queue, event_stop))
    threadCam.start()

    sensor0_data = sensor1_data = sensor2_data = sensorCam_frame = None
    logging.info(f'Finished initialisation at {time.strftime("%Y-%m-%d %H:%M:%S")}.')
    while not event_stop.is_set():
        # Ожидаем данные с камеры и сенсоров и отображаем их в окне
        if not thread0_queue.empty():
            sensor0_data = thread0_queue.get()
        if not thread1_queue.empty():
            sensor1_data = thread1_queue.get()
        if not thread2_queue.empty():
            sensor2_data = thread2_queue.get()
        if not threadCam_queue.empty():
            sensorCam_frame = threadCam_queue.get()

        window.show(sensor0_data, sensor1_data, sensor2_data, sensorCam_frame)

        # По нажатии 'q', выключаем камеру и окно, и сеттим сигнал
        # для функции worker для завершения потоков (иначе потоки
        # никогда не завершат работу и join() никогда не сработает).
        if cv2.waitKey(1) & 0xFF == ord('q'):
            event_stop.set()
            thread0.join()
            thread1.join()
            thread2.join()
            threadCam.join()

            window.close()
            sensorCam.close()
            logging.info(f'Stopped by user at {time.strftime("%Y-%m-%d %H:%M:%S")}.')

        time.sleep(cam_fps)

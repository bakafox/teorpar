import time
import cv2
import numpy as np
import argparse

import threading
import queue
from ultralytics import YOLO


def process_frame(model, frame):
    #model.to('cpu')
    result = model(frame)
    return result[0].plot()

def worker(frame_queue, result_queue):
    #model = YOLO('yolov8n-pose')
    model = YOLO('yolov8s-pose')

    while True:
        data = frame_queue.get()

        if data is None:
            break

        frame, idx = data
        result = process_frame(model, frame)
        result_queue.put((idx, result))
        frame_queue.task_done()


def process_video(input_path, output_path, max_threads):
    timestamp = time.time()

    VC = cv2.VideoCapture(input_path)
    frame_width = int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = VC.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    VW = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f'Thread count: {max_threads}')

    # Создаём и запускаем max_threads потоков с общими очередями
    frame_queue = queue.Queue()
    result_queue = queue.Queue()
    threads = []

    for i in range(max_threads):
        thread = threading.Thread(target=worker, args=(frame_queue, result_queue))
        thread.start()
        threads.append(thread)

    # Потоки запущены, кидаем в очередь все кадры оргинального видео
    frame_cnt = 0
    while VC.isOpened():
        ret, frame = VC.read()

        if not ret:
            break

        frame_queue.put((frame, frame_cnt))
        frame_cnt += 1
    VC.release()

    # Кидаем в очередь max_threads элекментов None;
    # в worker потоков для таких случаев стоит break,
    # т.е. поток завершит работу, когда дойдёт до него.
    for i in range(max_threads):
        frame_queue.put(None)

    # Ожидаем завершения работы всех потоков...
    for thread in threads:
        thread.join()

    # Наконец, создаём видео на основе результатов
    results = {}
    while not result_queue.empty():
        idx, result = result_queue.get()
        results[idx] = result

    for idx in range(frame_cnt):
        VW.write(results[idx])
    VW.release()

    print(f'Finished processing at {time.time() - timestamp} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.mp4')
    parser.add_argument('--output', type=str, default='output.mp4')
    parser.add_argument('--use_mt', type=int, default=1)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    max_threads = 1 if (args.use_mt == 0) else 20 # 16-20 оптимально

    process_video(input_path, output_path, max_threads)

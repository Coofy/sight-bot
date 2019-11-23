import time

import cv2
import mss
import numpy as np
from PIL import Image

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pyautogui
from pynput import mouse


def load_category_index(path):
    category_index = label_map_util.create_category_index_from_labelmap(
        path, use_display_name=True)

    return category_index


def load_frozen_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def run_inference(sess, detection_graph, image_np):
    image_expanded_np = np.expand_dims(image_np, 0)

    with detection_graph.as_default():
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name(
            'num_detections:0')

        start_time = time.time()
        (boxes, scores, classes, num_detections) = \
            sess.run([boxes, scores, classes, num_detections],
                     feed_dict={image_tensor: image_expanded_np})
        end_time = time.time()
        #print("Inference time: %d" % (int((end_time-start_time) * 1000)))

        return (boxes, scores, classes, num_detections)


def draw_inference_result(image_np, category_index, inference_result):
    (boxes, scores, classes, num_detections) = inference_result

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)

    return image_np


can_aim = []
def on_click(x, y, button, pressed):
    if button == mouse.Button.right and pressed:
        can_aim.append(True)


def aim(inference_result, windowRect):
    print('-----------------------------------')
    biggestScore = (None, 0, None, None)
    n = len(inference_result[0])
    for i in range(n):
        if inference_result[1][0][i] > biggestScore[1]:
            biggestScore = (inference_result[0][i], inference_result[1][i],
            inference_result[2][i], inference_result[3][i])
    
    print("Biggest: ", biggestScore[1][0])
    print("Biggest: ", biggestScore[0][0])

    targetX = (biggestScore[0][0][1] + biggestScore[0][0][3]) / 2 - 0.5
    targetY = (biggestScore[0][0][0] + biggestScore[0][0][2]) / 2 - 0.5

    targetX = targetX * windowRect['width']
    constX = 100
    if targetX >= 0:
        targetX += constX
    else:
        targetX -= constX

    pyautogui.moveTo(targetX, 0)


print("Hello, World!")

pyautogui.FAILSAFE = False

category_index = load_category_index('workspace/training_demo/annotations/label_map.pbtxt')
detection_graph = load_frozen_graph('workspace/training_demo/trained-inference-graphs/output_inference_graph_v1.pb/frozen_inference_graph.pb')

windowRect = {"top": 60, "left": 68, "width": 1024, "height": 768}

listener = mouse.Listener(on_click=on_click)
listener.start()

# Reuse session
with tf.compat.v1.Session(graph=detection_graph) as sess:
    with mss.mss() as sct:
        while(True):
            start_time = time.time()
            frame = np.array(sct.grab(windowRect))
            end_time = time.time()
            #print("Grab time: %d" % (int((end_time-start_time) * 1000)))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            inference_result = run_inference(sess, detection_graph, frame)
            if len(can_aim) > 0:
                aim(inference_result, windowRect)
                can_aim.clear()
            image_result_np = draw_inference_result(frame, category_index,
                                                    inference_result)

            cv2.imshow('Frame', cv2.cvtColor(image_result_np, cv2.COLOR_BGR2RGB))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
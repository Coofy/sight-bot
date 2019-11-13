import cv2
import numpy as np
from PIL import Image

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_category_index(path):
    label_map = label_map_util.load_labelmap(path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, 4, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

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


def run_inference(detection_graph, image_np):
    image_expanded_np = np.expand_dims(image_np, axis=0)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            (boxes, scores, classes, num_detections) = \
                sess.run([boxes, scores, classes, num_detections],
                         feed_dict={image_tensor: image_expanded_np})

            return (boxes, scores, classes, num_detections)


def draw_inference_result(image, category_index, inference_result):
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


print("Hello, World!")

category_index = load_category_index('labelmap.pbtxt')
detection_graph = load_frozen_graph('frozen_inference_graph.pb')

image_np = np.array(Image.open("frame.jpg"))
inference_result = run_inference(detection_graph, image_np)
image_result_np = draw_inference_result(image_np, category_index,
                                        inference_result)

cv2.imshow('Result', cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)

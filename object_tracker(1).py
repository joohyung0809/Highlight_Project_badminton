# 처음 진행하는 것. output_second.csv , C:/Users/joohy/Computer_Vision/yolov4-deepsort/outputs/from_object_tracker(1).mp4

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import fileinput
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import csv


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/data/joo_5min.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/outputs/from_object_tracker(1).mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.8, 'iou threshold') # default:0.45
flags.DEFINE_float('score', 0.8, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def is_person_wearing_black_color(image_path, target_color, color_range):
    # 이미지를 불러오기
    image = image_path
    
    # 이미지에서 특정 색상 범위에 해당하는 픽셀 찾기
    lower_color = np.array([max(0, c - color_range) for c in target_color], dtype=np.uint8)
    upper_color = np.array([min(255, c + color_range) for c in target_color], dtype=np.uint8)
    
    # 이미지에서 색상 범위에 해당하는 픽셀을 찾음
    color_mask = cv2.inRange(image, lower_color, upper_color)

    # 해당 색상을 입은 픽셀의 개수를 세기
    color_pixel_count = np.sum(color_mask > 0)

    # 픽셀 개수를 기준으로 해당 색상의 옷을 입었는지 여부 판단
    threshold_pixel_count = 4000  # 적절한 임계값 설정
    return color_pixel_count > threshold_pixel_count

def is_person_wearing_color(image_path, target_color, color_range):
    # 이미지를 불러오기
    image = image_path
    
    # print("Image shape:", image.shape)

    # 이미지에서 특정 색상 범위에 해당하는 픽셀 찾기
    lower_color = np.array([max(0, c - color_range) for c in target_color], dtype=np.uint8)
    upper_color = np.array([min(255, c + color_range) for c in target_color], dtype=np.uint8)
    
    # 이미지에서 색상 범위에 해당하는 픽셀을 찾음
    color_mask = cv2.inRange(image, lower_color, upper_color)

    # 해당 색상을 입은 픽셀의 개수를 세기
    color_pixel_count = np.sum(color_mask > 0)

    # 픽셀 개수를 기준으로 해당 색상의 옷을 입었는지 여부 판단
    threshold_pixel_count = 1800  # 적절한 임계값 설정
    return color_pixel_count > threshold_pixel_count

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.6 # 클 수록 다른 객체로 인식하는 비율 줄어듦
    nn_budget = 400
    nms_max_overlap = 1.0
    disappear_moment = [] # 해당 객체 id가 사라진 모든 프레임 no
    target_id = 1
    output_captures_path = 'C:/Users/joohy/Computer_Vision/yolov4-deepsort/dupli_id_one/'
    
    
    # joohyung clothe's info
    target_shirt_color = [122, 109, 116]
    shirt_color_range = 20

    target_pants_color = [62, 54, 58]
    pants_color_range = 20

    target_black_color = [14, 14, 20]
    black_color_range = 20
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    
    # For csv output
    csv_file_path = './save_file/output_second.csv'
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Object ID','Class' ,'xmin', 'ymin', 'xmax', 'ymax'])

    prev_frame_rgb = None

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        
        center_x = frame.shape[1]//2
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        one_frame_id=one_frame_id = [track.track_id for track in tracker.tracks if track.is_confirmed() and track.time_since_update <= 1]
        if target_id not in one_frame_id:
            disappear_moment.append(frame_num)
            
        
        # Calculate RGB difference between current and previous frame
        frame_rgb = np.mean(frame, axis=(0, 1))
        if prev_frame_rgb is not None:
            rgb_difference = frame_rgb - prev_frame_rgb
            for i, diff in enumerate(rgb_difference):
                target_shirt_color[i] += diff
                target_pants_color[i] += diff
                target_black_color[i] += diff
            print("RGB Difference:", rgb_difference)
            print(target_shirt_color)
            print(target_pants_color)
            print(target_black_color)
            
    
        # Update previous frame RGB values
        prev_frame_rgb = frame_rgb.copy()
        
        
        for i, track in enumerate(tracker.tracks):
            
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
                
            if track.track_id == target_id: # target_id가 있다는 건 disappear_moment에 추가 안 된 frame
                if (int(bbox[1]) + int(bbox[0])) / 2 > (width / 2):
                    track.track_id = 999
            
            center_x_bbox = (bbox[0] + bbox[2]) // 2
            
            if frame_num in disappear_moment: # target_id가 없는 프레임의 클래스 전부 들어감
                bbox = [max(0, int(coord)) for coord in bbox]
                # if xmin < 0 or ymax > original_h:
                #     print(f'Object with ID {track.track_id} has exited the video at frame {frame_num}')
                # else:
                #     print(f'Object with ID {track.track_id} has not exited the video at frame {frame_num}')
                captured_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                # image_filename = f'./save_file/image/captured_image_{frame_num}_{track.track_id}.png'
                
                # target_id가 아닌 id들이 joohyung인지 체크
                if (is_person_wearing_color(captured_image, target_shirt_color, shirt_color_range) and is_person_wearing_color(captured_image, target_pants_color, pants_color_range)) and not is_person_wearing_black_color(captured_image, target_black_color, black_color_range)  and (center_x_bbox < center_x):
                # if not is_person_wearing_black_color(captured_image, target_black_color, black_color_range)  and (center_x_bbox < center_x):    
                    # print(f'captured_image_{frame_num}_{track.track_id}is joohyung') 
                    
                    # track.track_id = new_object_id
                    
                    # Write information to CSV file
                    new_object_id = 1
                    csv_writer.writerow([frame_num, new_object_id, class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    
                    # data = pd.read_csv("./save_file/output.csv")
                    # row_index = data[(data['Frame'] == frame_num) & (data['Object ID'] == track.track_id)].index
                    
                    # new_object_id = target_id
                    # data.at[row_index, 'Object ID'] = new_object_id
                    
                    # data.to_csv('./save_file/output.csv', index=False)
                    # continue
                else:
                    csv_writer.writerow([frame_num, track.track_id, class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                    # cv2.imwrite(image_filename, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
            else:
                # Write information to CSV file
                csv_writer.writerow([frame_num, track.track_id, class_name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                
            # draw bbox on screen -> 원래 999처리 위에 있었음
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            # xmin, ymin, xmax, ymax = bbox
            
            #     else:
            #         captured_dupli_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            #         capture_path = os.path.join(output_captures_path, f"capture_frame_{frame_num}_id_{target_id}.png")
            #         cv2.imwrite(capture_path, captured_dupli_image, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
            #         print(f"Captured Frame: {frame_num}, Object ID: {target_id}. Saved at: {capture_path}")
            
                
            
            
            # one_frame_id.append(track.track_id)
            
            
            
            # if(track.track_id != 3):
            #     if frame_num not in disappear_moment:
            #         disappear_moment.append(frame_num)
                
                
            # if frame_num in disappear_moment:
            #     # Save the image with bounding box
            #     captured_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            #     # cv2.imwrite(f'captured_image_{frame_num}_{track.track_id}.png', cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))
            #     if is_person_wearing_color(captured_image, target_shirt_color, shirt_color_range) and is_person_wearing_color(captured_image, target_pants_color, pants_color_range):
            #         print(f'captured_image_{frame_num}_{track.track_id}is joohyung')
                    



            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            

        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    file_path = './save_file/disappear_moment.txt'
    np.savetxt(file_path, disappear_moment, fmt='%d', delimiter=', ')
    
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
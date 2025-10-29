from ultralytics import YOLO #type: ignore
import cv2
from utils import get_car, read_license_plate
import pandas as pd

car_detector_model = YOLO('./models/yolo11n.pt')
license_plate_detector = YOLO('./models/license_plate_model_best.pt')
vehicles_class_indexes = [2, 3, 5, 7]
cap = cv2.VideoCapture('./videos/traffic.mp4')
frame_count = -1
results = []
while True:
    frame_count += 1
    print("Frame: ", frame_count)
    ret, frame = cap.read()
    if not ret:
        break
    detections = car_detector_model.track(frame, persist=True)[0]
    vehicle_detections = []
    for detection in detections.boxes.data.tolist(): # type: ignore
        x1, y1, x2, y2, vehicle_detection_id, vehicle_confidence_score, class_id = detection
        if int(class_id) in vehicles_class_indexes:
            vehicle_detections.append([x1, y1, x2, y2, vehicle_detection_id, vehicle_confidence_score, class_id])
    license_plate_detections = license_plate_detector.predict(frame)[0]
    for license_plate in license_plate_detections.boxes.data.tolist(): # type: ignore
        x1, y1, x2, y2, plate_confidence_score, class_id = license_plate
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, track_id, confidence_score, class_id = get_car(license_plate, vehicle_detections)
        if track_id == -1:
            break
        licence_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY)
        _, licence_plate_crop_thresh = cv2.threshold(licence_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        licence_plate_text, licence_plate_text_score = read_license_plate(licence_plate_crop_thresh)
        if licence_plate_text is not None:
            new_results = {'frame': frame_count, 
                                     'track_id': track_id,
                                     'vehicle_bbox': [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2],
                                     'license_plate_bbox': [x1, y1, x2, y2],
                                     'text': licence_plate_text,
                                     'license_plate_bbox_score': plate_confidence_score,
                                     'plate_text_score': licence_plate_text_score}
            results.append(new_results)

results_dataframe = pd.DataFrame(results)
results_dataframe.to_excel('./outputs/results.xlsx', index=False)
cap.release()    
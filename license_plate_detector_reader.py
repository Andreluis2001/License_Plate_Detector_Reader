from ultralytics import YOLO #type: ignore
import cv2
from utils import get_car, read_license_plate
from interpolate_missing_data import interpolate_bounding_boxes
import pandas as pd

class LicensePlateDetectorandReader:
    def __init__(
            self, 
            car_detector_model_path, 
            license_plate_detector_model_path, 
            vehicle_target_indexes
    ):
        self.car_detector_model = YOLO(car_detector_model_path)
        self.license_plate_detector = YOLO(license_plate_detector_model_path)
        self.target_vehicle_indexes = vehicle_target_indexes

    def detect_vehicles(self, frame):
        vehicle_detections = []
        detections = self.car_detector_model.track(frame, persist=True)[0]
        for detection in detections.boxes.data.tolist(): #type: ignore
            x1, y1, x2, y2, vehicle_detection_id, vehicle_confidence_score, class_id = detection
            if int(class_id) in self.target_vehicle_indexes:
                vehicle_detections.append([x1, y1, x2, y2, vehicle_detection_id, vehicle_confidence_score, class_id])
        return vehicle_detections
    
    def detect_license_plates(self, frame):
        results = []
        vehicle_detections = self.detect_vehicles(frame)
        detections = self.license_plate_detector.predict(frame)[0]
        for license_plate in detections.boxes.data.tolist(): # type: ignore
            x1, y1, x2, y2, plate_confidence_score, class_id = license_plate
            vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, track_id, vehicle_confidence_score, class_id = get_car(license_plate, vehicle_detections)
            if track_id == -1:
                break
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                new_results = [
                    track_id,
                    [vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2],
                    [x1, y1, x2, y2],
                    class_id,
                    vehicle_confidence_score,
                    license_plate_text,
                    plate_confidence_score,
                    license_plate_text_score
                ]
                results.append(new_results)
        return results
    
    def perform_detections(self, video_path, save=False, interpolate_bboxes=False, save_path=None):
        cap = cv2.VideoCapture(video_path)
        frame_count = -1
        results = []
        while True:
            frame_count += 1
            ret, frame = cap.read()
            if not ret:
                break
            license_plate_detections = self.detect_license_plates(frame)
            for detection in license_plate_detections:
                new_results = {'frame': frame_count, 
                                     'track_id': detection[0],
                                     'vehicle_bbox': detection[1],
                                     'license_plate_bbox': detection[2],
                                     'vehicle_class_id': detection[3],
                                     'vehicle_bbox_score': detection[4],
                                     'license_plate_number': detection[5],
                                     'license_plate_bbox_score': detection[6],
                                     'license_plate_text_score': detection[7]}
                results.append(new_results)
        results_df = pd.DataFrame(results)
        if interpolate_bboxes:
            results_df = interpolate_bounding_boxes(results_df)
        if save:
            self.save_results(results_df, save_path)
        return results_df
    
    def save_results(self, results_df, save_path):
        if save_path.split('.')[-1] == 'csv':
            results_df.to_csv(save_path, index=False)
        elif save_path.split('.')[-1] == 'xlsx':
            results_df.to_excel(save_path, index=False)
        elif save_path.split('.')[-1] == 'json':
            results_df.to_json(save_path, orient='records')
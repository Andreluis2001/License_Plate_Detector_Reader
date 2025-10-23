from ultralytics import YOLO #type: ignore

car_detection_model = YOLO('./models/yolo11n.pt')
results = car_detection_model.track('./videos/short_traffic.mp4', save=True)
vehicle_detections = []
for box in results[0].boxes: #type: ignore
    print(box.data.tolist()[0])
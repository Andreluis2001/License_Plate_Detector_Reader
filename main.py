from license_plate_detector_reader import LicensePlateDetectorandReader #type: ignore

detector = LicensePlateDetectorandReader( 
                                            './models/yolo11n.pt', 
                                            './models/license_plate_model_best.pt', 
                                            [2, 3, 5, 7]
                                        )

detector.perform_detections(
    './videos/short_traffic.mp4', 
    save=True, 
    interpolate_bboxes=True,
    save_path='./outputs/results_short.xlsx'
    )
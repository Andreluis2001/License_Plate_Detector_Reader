import license_plate_detector_reader

detector = license_plate_detector_reader.LicensePlateDetectorandReader( # type: ignore
                                            './models/yolo11n.pt', 
                                            './models/license_plate_model_best.pt', 
                                            [2, 3, 5, 7]
                                        )

detector.perform_detections('./videos/short_traffic.mp4', save=True, save_path='./outputs/results.xlsx')
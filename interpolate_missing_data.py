import csv 
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    interpolated_data = []
    frame_numbers = np.array([int(row['frame']) for row in data])
    tracks_ids = np.array([int(row['track_id']) for row in data])
    vehicles_bboxes = np.array([row['vehicle_bbox'] for row in data])
    license_plate_bboxes = np.array([row['license_plate_bbox']] for row in data)
    unique_vehicle_ids = np.unique(tracks_ids)
    for vehicle_id in unique_vehicle_ids:
        frames = [row['frame'] for row in data if int(row['track_id']) == int(vehicle_id)]
        vehicle_mask = tracks_ids == vehicle_id
        first_frame = frames[0]
        last_frame = frames[-1]
        interpolated_vehicle_bboxes = []
        interpolated_license_plate_bboxes = []
        for i in range(len(vehicles_bboxes[vehicle_mask])):
            current_vehicle_frame = frames[i]
            current_vehicle_bbox = vehicles_bboxes[vehicle_mask][i]
            current_license_plate_bbox = license_plate_bboxes[vehicle_mask][i]
            if i > 0:
                prev_frame = frames[i-1]
                prev_vehicle_bbox = interpolated_vehicle_bboxes[-1]
                prev_license_plate_bbox = interpolated_license_plate_bboxes[-1]
                if current_vehicle_frame - prev_frame > 1:
                    frame_gap = current_vehicle_frame - prev_frame
                    x = np.array([prev_frame, current_vehicle_frame])
                    x_new = np.linspace(prev_frame, current_vehicle_frame, num=frame_gap, endpoint=False)
                    new_vehicle_bboxes = np.interp(x_new, x, (prev_vehicle_bbox, current_vehicle_bbox))
                    new_license_plate_bboxes = np.interp(x_new, x, (prev_license_plate_bbox, current_license_plate_bbox))
                    interpolated_vehicle_bboxes.extend(new_vehicle_bboxes[1:])
                    interpolated_license_plate_bboxes.extend(new_license_plate_bboxes[1:])
            interpolated_vehicle_bboxes.append(current_vehicle_bbox)
            interpolated_license_plate_bboxes.append(current_license_plate_bbox)
        for i in range(len(interpolated_vehicle_bboxes)):
            vehicle_frame = first_frame + 1
            new_row = {}
            new_row['frame'] = vehicle_frame
            new_row['track_id'] = vehicle_id
            new_row['vehicle_bbox'] = interpolated_vehicle_bboxes[i]
            new_row['license_plate_bbox'] = interpolated_license_plate_bboxes[i]

            if vehicle_frame not in frames:
                new_row['text'] = '0000000'
                new_row['license_plate_bbox_score'] = 0.0
                new_row['plate_text_score'] = 0.0
            else:
                original_row = [row for row in data if int(row['frame']) == vehicle_frame and int(row['track_id']) == int(vehicle_id)][0]
                new_row['text'] = original_row['text'] if 'text' in original_row else '0000000'
                new_row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else 0.0
                new_row['plate_text_score'] = original_row['plate_text_score'] if 'plate_text_score' in original_row else 0.0
            interpolated_data.append(new_row)
    return interpolated_data
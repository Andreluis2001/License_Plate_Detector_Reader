import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def get_interpolation_function(x_new, x, y):
    """
    Get interpolation function using Scipy's interp1d for 1d values interpolation
    
    Args:
        x_new (list(float)): List of x values to create interpolated y values.
        x (list(float)): Interval of x values.
        y (list(float)): Interval of y values.
    
    Return: 
        list(float): Interpolated y values.
    """
    interp_func = interp1d(
                    x, 
                    np.vstack(y),
                    axis=0,
                    kind='linear'
    )

    return interp_func(x_new)

def interpolate_bounding_boxes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates bounding boxes for the missing frames for each vehicle being tracked.

    Args:
        dataframe (pd.DataFrame): Dataframe containing vehicle and license plate detections.

    Return:
        New pandas dataframe with interpolated values.
    """
    dataframe = dataframe.drop_duplicates(subset=['frame', 'track_id'])
    interpolated_data = []
    track_ids = np.array(dataframe['track_id'])

    # Loop through each unique vehicle tracking id
    for vehicle_id in np.unique(track_ids):
        frames = np.array(dataframe.loc[dataframe['track_id'] == vehicle_id, 'frame'])
        current_vehicle_bboxes = dataframe.loc[dataframe['track_id'] == vehicle_id, 'vehicle_bbox'].to_list() # type: ignore
        current_license_plate_bboxes = dataframe.loc[dataframe['track_id'] == vehicle_id, 'license_plate_bbox'].to_list() # type: ignore

        interpolated_vehicle_bboxes = []
        interpolated_license_plate_bboxes = []

        # Loop through each frame for each vehicle tracking id 
        for i in range(len(frames)):
            current_frame = frames[i]
            current_vehicle_bbox = current_vehicle_bboxes[i] # type: ignore
            current_license_plate_bbox = current_license_plate_bboxes[i]
            # Ignore first frame
            if i > 0:
                prev_frame = frames[i-1]
                prev_vehicle_bbox = interpolated_vehicle_bboxes[-1]
                prev_license_plate_bbox = interpolated_license_plate_bboxes[-1]

                # Ignore sucessive frames
                if current_frame - prev_frame > 1:
                    frame_gap = current_frame - prev_frame
                    x = np.array([prev_frame, current_frame])
                    x_new = np.linspace(prev_frame, current_frame, num=frame_gap, endpoint=False)
                    new_vehicle_bboxes = get_interpolation_function(x_new, x, (prev_vehicle_bbox, current_vehicle_bbox))
                    new_license_plate_bboxes = get_interpolation_function(x_new, x, (prev_license_plate_bbox, current_license_plate_bbox))
                    interpolated_vehicle_bboxes.extend(new_vehicle_bboxes)
                    interpolated_license_plate_bboxes.extend(new_license_plate_bboxes)

            interpolated_vehicle_bboxes.append(np.array(current_vehicle_bbox))
            interpolated_license_plate_bboxes.append(np.array(current_license_plate_bbox))

        # Add interpolated rows to dataframe
        for i in range(len(interpolated_vehicle_bboxes)):
            vehicle_frame = frames[0] + i
            new_row = {}
            new_row['frame'] = vehicle_frame
            new_row['track_id'] = vehicle_id
            new_row['vehicle_bbox'] = interpolated_vehicle_bboxes[i]
            new_row['license_plate_bbox'] = interpolated_license_plate_bboxes[i]
            if vehicle_frame in frames:
                original_row = dataframe.loc[
                    (dataframe['frame'] == vehicle_frame) & (dataframe['track_id'] == vehicle_id)
                ]
                new_row['vehicle_class_id'] = original_row['vehicle_class_id'].iloc[0]
                new_row['vehicle_bbox_score'] = original_row['vehicle_bbox_score'].iloc[0]
                new_row['license_plate_number'] = original_row['license_plate_number'].iloc[0]
                new_row['license_plate_text_score'] = original_row['license_plate_text_score'].iloc[0]
            else:
                new_row['vehicle_class_id'] = None
                new_row['vehicle_bbox_score'] = None
                new_row['license_plate_number'] = None
                new_row['license_plate_text_score'] = None
            interpolated_data.append(new_row)

    return pd.DataFrame(interpolated_data)
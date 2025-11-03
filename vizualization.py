import cv2

def draw_border(
        img, 
        top_left_corner, 
        bottom_right_corner, 
        color=(255, 0, 0),
        thickness=10
):
    cv2.rectangle(
        img=img, 
        pt1=top_left_corner, 
        pt2=bottom_right_corner, 
        color=color, 
        thickness=thickness
    )

    return img

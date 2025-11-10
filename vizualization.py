import cv2
import pandas as pd

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

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_excel('./outputs/results_short.xlsx')
    print(df)
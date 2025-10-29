import easyocr
import string

reader = easyocr.Reader(['en'], gpu=False)

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

dict_char_to_int = {v: k for k, v in dict_int_to_char.items()}

def check_licence_reading_complies(reading):
    if len(reading) != 7:
        return False
    for i in [0,1,4,5,6]:
        if reading[i] not in string.ascii_uppercase and reading[i] not in dict_int_to_char.keys():
            return False
    for i in [2,3]:
        if reading[i] not in [str(i) for i in range(10)] and reading[i] not in dict_char_to_int.keys():
            return False 
    return True

def format_plate_reading(reading):

    license_plate = ''
    for i in range(7):
        if i in [2, 3]:
            if reading[i] in dict_char_to_int.keys():
                license_plate += dict_char_to_int[reading[i]]
            else:
                license_plate += reading[i]
        else:
            if reading[i] in dict_int_to_char.keys():
                license_plate += dict_int_to_char[reading[i]]
            else:
                license_plate += reading[i]
    return license_plate


def read_license_plate(license_plate_image):
    readings = reader.readtext(license_plate_image)
    for reading in readings:
        bbox, text, score = reading
        text = text.replace(' ', '')
        if check_licence_reading_complies(text):
            return format_plate_reading(text), score
    return None, None


def get_car(license_plate, vehicles):
    x1, y1, x2, y2, score, class_id = license_plate
    vehicle_found = False
    vehicle_id = 0
    for v in range(len(vehicles)):
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2, track_id, score, class_id = vehicles[v]
        if x1 > vehicle_x1 and y1 > vehicle_y1 and x2 < vehicle_x2 and y2 < vehicle_y2:
            vehicle_id = v
            vehicle_found = True
            break
    if vehicle_found:
        return vehicles[vehicle_id]
    
    return -1, -1, -1, -1, -1, -1, -1
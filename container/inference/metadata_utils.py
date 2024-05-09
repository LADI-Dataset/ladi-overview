import os

from PIL import Image, ImageFile
from PIL.ExifTags import TAGS, GPSTAGS
from multiprocessing.pool import ThreadPool
import numpy as np
import pathlib
import tqdm
import argparse
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True

# here's some functions to parse the exif data and convert them to decimal format
def parse_exif(exif_dict):
    human_readable = {}
    if exif_dict == None:
        return None
    for k, v in exif_dict.items():
        try:
            text_key = TAGS[k]
            # if type(v) == tuple and len(v)==2:
            #     v = divide_tuple(v)
            if type(v) == bytes:
                v = v.hex()
            if k == 34853: # GPS info field
                human_readable[text_key] = parse_gps(v)
            elif not ((text_key == 'UserComment') or (text_key == 'MakerNote')):
                human_readable[text_key] = v
        except:
            pass
    return human_readable

def parse_gps(gps_dict):
    readable_gps = {}
    for k,v in gps_dict.items():
        try:
            text_key = GPSTAGS[k]
            if text_key in ['GPSLatitude', 'GPSLongitude']:
                # print(v)
                v = convert_GPS_coord(v)
            if type(v) == bytes:
                v = v.hex()
            readable_gps[text_key] = v
        except:
            pass
    return readable_gps

def convert_GPS_coord(GPS_tuple):
    """ Converts the GPS coordinate given by exif into decimal coord
    GPS_Tuple = (
       degrees, minutes, seconds
       )
    """
    try:
        degs = float(GPS_tuple[0])
        mins = float(GPS_tuple[1])
        secs = float(GPS_tuple[2])
        return degs + mins/60 + secs/3600
    except:
        return None

def get_lat_lon(exif):
    try:
        lat = exif['GPSInfo']['GPSLatitude']
        if exif['GPSInfo']['GPSLatitudeRef']=='S':
            lat *= -1
    except:
        lat = None
    try:
        lon = exif['GPSInfo']['GPSLongitude']
        if exif['GPSInfo']['GPSLongitudeRef']=='W':
            lon *= -1
    except:
        lon = None
    return lat, lon

def get_metadata_img(img):
    output = {}
    exif_dict = parse_exif(img._getexif())
    if exif_dict == None:
        output['lat'], output['lon'], output['timestamp'] = (None, None, None)
    else:
        output['lat'], output['lon'] = get_lat_lon(exif_dict)
        try:
            output['timestamp'] = exif_dict['DateTimeOriginal']
        except KeyError:
            output['timestamp'] = None
    return output

def get_metadata_entry(img_path):
    try:
        img = Image.open(img_path)
    except:
        return
    return {'file_path': img_path, **get_metadata_img(img)}


if __name__=="__main__":
    with open('./file_list.txt','r') as img_path_file:
        img_paths_list = img_path_file.read().splitlines()
    img_paths_list = [x for x in img_paths_list if x.split('.')[-1].lower() in ['png','jpg','jpeg']]
    worker_img_paths_list = img_paths_list
    with ThreadPool(os.cpu_count()) as p:
        output_list = p.map(get_metadata_entry, worker_img_paths_list)
    output_list = [x for x in output_list if x is not None]
    output_df = pd.DataFrame(output_list)
    output_df = output_df.dropna()
    output_df = output_df.loc[output_df['timestamp']!='0000:00:00 00:00:00']
    output_df['timestamp'] = pd.to_datetime(output_df['timestamp'], format='%Y:%m:%d %H:%M:%S')
    output_df.to_csv(f'outputs.csv', index=False)


#!/usr/bin/env python3
import json
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime as dt
import time
import argparse

parser = argparse.ArgumentParser(description = "Pull NYC DOT traffic webcam pictures")
parser.add_argument("cams_json_file_loc", nargs = "?", default = "./cams.json")
parser.add_argument("log_file_loc", nargs = "?", default = "./pics.log")
parser.add_argument("pic_folder", nargs = "?", default = "./pics/")
args = parser.parse_args()

# python3 pull_pics.py ./cams.json ./pics.log /mnt/disks/strge/nyc_dot_webcam_pics/

# cams_json_file_loc = "./cams.json"
# log_file_loc = "./pics.log"
# pic_folder = "./pics/"

def get_pic(cams_json_file_loc, log_file_loc, pic_folder, prnt = False):
    # get all cams
    with open(cams_json_file_loc, "r") as f:
        cams = json.load(f)

    cam_ids = [cam["cam_id"] for cam in cams]

    # read the log to see which cam was last pulled
    curr_cam_id = cam_ids[0]
    with open(log_file_loc, "r") as f:
        lines = f.readlines()
        if len(lines) > 0:
            last_line = lines[-1]
            last_cam_id = int(last_line.split("\t")[0].split(" ")[1])
            last_cam_id_ind = [i for i, cam_id in enumerate(cam_ids) if cam_id == last_cam_id][0]
            curr_cam_id_ind = last_cam_id_ind + 1 if last_cam_id_ind + 1 < len(cam_ids) else 0
            curr_cam_id = cam_ids[curr_cam_id_ind]

    curr_cam = cams[curr_cam_id]

    # get the image
    pull_dt = dt.now().strftime("%m%d%y_%H%M%S")
    img_addr = curr_cam["img_addr"]
    status = "pulled"
    img_filename = ""
    try:
        raw_img = requests.get(img_addr).content
        img = Image.open(BytesIO(raw_img))

        # save the image
        img_filename = "cam_id_{}_{}.png".format(curr_cam_id, pull_dt)
        img.save("{}{}".format(pic_folder, img_filename))

    except:
        status = "failed"


    # update the log
    msg = "cam_id: {} pulled: {} to: {}{} status: {}\n".format(curr_cam_id, pull_dt, pic_folder, img_filename, status)
    with open(log_file_loc, "a") as f:
        f.write(msg)
        if prnt:
            print(msg[:-2])

for i in range(60):
    get_pic(
        cams_json_file_loc = args.cams_json_file_loc,
        log_file_loc = args.log_file_loc,
        pic_folder = args.pic_folder,
        prnt = True)
    time.sleep(1)

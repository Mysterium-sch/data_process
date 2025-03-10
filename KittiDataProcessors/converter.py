#!/usr/bin/env python2

from PIL import Image
import argparse
import os
import csv
from pathlib import Path

OUT_LABELS_DIR = "labels"

KEY_PEDESTRIAN = "Pedestrian"
KEY_CYCLIST = "Cyclist"
KEY_CAR = "Car"
KEY_VAN = "Van"
KEY_MISC = "Misc"
KEY_TRUCK = "Truck"
KEY_PERSON_SITTING = "Person_sitting"
KEY_TRAM = "Tram"
KEY_DONT_CARE = "DontCare"

CLAZZ_NUMBERS = {
            KEY_PEDESTRIAN : 0,
            KEY_CYCLIST : 1,
            KEY_CAR : 2,
            KEY_DONT_CARE : 3
        }

def getSampleId(path):
    basename = os.path.basename(path)
    return os.path.splitext(basename)[0]

def resolveClazzNumberOrNone(clazz, use_dont_care):
    if clazz == KEY_CYCLIST:
        return CLAZZ_NUMBERS[KEY_CYCLIST]
    #if clazz in (KEY_PEDESTRIAN, KEY_PERSON_SITTING):
    if clazz == KEY_PEDESTRIAN:
        return CLAZZ_NUMBERS[KEY_PEDESTRIAN]
    #if clazz in (KEY_CAR, KEY_VAN):
    if clazz == KEY_CAR:
        return CLAZZ_NUMBERS[KEY_CAR]
    if use_dont_care and clazz == KEY_DONT_CARE:
        return CLAZZ_NUMBERS[KEY_DONT_CARE] # These should not be punished.
    return None

def convertToYoloBBox(bbox, size):
    # Yolo uses bounding bbox coordinates and size relative to the image size.
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[1]) / 2.0
    y = (bbox[2] + bbox[3]) / 2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    
    # Normalize the coordinates
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    # Clamp values to be within [0, 1]
    x = max(0, min(x, 1))
    y = max(0, min(y, 1))
    w = max(0, min(w, 1))
    h = max(0, min(h, 1))
    
    return (x, y, w, h)

def readRealImageSize(img_path):
    # This loads the whole sample image and returns its size.
    return Image.open(img_path).size

def readFixedImageSize():
    # This is not exact for all images but most (and it should be faster).
    return (1242, 375)

def parseSample(lbl_path, img_path, use_dont_care):
    with open(lbl_path) as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=["type", "truncated", "occluded", "alpha", "bbox2_left", "bbox2_top", "bbox2_right", "bbox2_bottom", "bbox3_height", "bbox3_width", "bbox3_length", "bbox3_x", "bbox3_y", "bbox3_z", "bbox3_yaw", "score"], delimiter=" ")
        yolo_labels = []
        for row in reader:
            clazz_number = resolveClazzNumberOrNone(row["type"], use_dont_care)
            if clazz_number is not None:
                size = readRealImageSize(img_path)
                #size = readFixedImageSize()
                # Image coordinate is in the top left corner.
                bbox = (
                        float(row["bbox2_left"]),
                        float(row["bbox2_right"]),
                        float(row["bbox2_top"]),
                        float(row["bbox2_bottom"])
                       )
                yolo_bbox = convertToYoloBBox(bbox, size)
                # Yolo expects the labels in the form:
                # <object-class> <x> <y> <width> <height>.
                yolo_label = (clazz_number,) + yolo_bbox
                yolo_labels.append(yolo_label)
    return yolo_labels

def parseArguments():
    parser = argparse.ArgumentParser(description="Generates labels for training darknet on KITTI.")
    parser.add_argument("label_dir", help="data_object_label_2/training/label_2 directory; can be downloaded from KITTI.")
    parser.add_argument("image_2_dir", help="data_object_image_2/training/image_2 directory; can be downloaded from KITTI.")
    parser.add_argument("lidar_2_dir", help="data_object_image_2/training/image_2 directory; can be downloaded from KITTI.")
    parser.add_argument("--training_samples", type=int, default=0.8, help="percentage of the samples to be used for training between 0.0 and 1.0.")
    parser.add_argument("--use-dont-care", action="store_true", help="do not ignore 'DontCare' labels.")
    args = parser.parse_args()
    if args.training_samples < 0 or args.training_samples > 1:
        print("Invalid argument {} for --training-samples. Expected a percentage value between 0.0 and 1.0.")
        exit(-1)
    return args

def main():
    args = parseArguments()

    if not os.path.exists(OUT_LABELS_DIR):
        os.makedirs(OUT_LABELS_DIR)

    print("Generating darknet labels...")
    sample_img_pathes = []
    for dir_path, sub_dirs, files in os.walk(args.label_dir):
        for file_name in files:
            if file_name.endswith(".txt"):
                lbl_path = os.path.join(dir_path, file_name)
                sample_id = getSampleId(lbl_path)
                img_path = os.path.join(args.image_2_dir, "{}.png".format(sample_id))
                sample_img_pathes.append(img_path)
                yolo_labels = parseSample(lbl_path, img_path, args.use_dont_care)
                with open(os.path.join(OUT_LABELS_DIR, "{}.txt".format(sample_id)), "w") as yolo_label_file:
                    for lbl in yolo_labels:
                        yolo_label_file.write("{} {} {} {} {}\n".format(*lbl))

    print("Writing training and test sample ids...")
    first_test_sample_index = int(args.training_samples * len(sample_img_pathes))
    with open("kitti_train.txt", "w") as train_file, open("kitti_test.txt", "w") as test_file:
        for sample_index in range(first_test_sample_index):
            # Writing image and lidar paths for training samples
            img_path = sample_img_pathes[sample_index]
            filename = Path(img_path).stem
            lidar_path = str(Path(args.lidar_2_dir, (str(filename) + ".npy"))) # Assuming lidar has the same base name as the image
            #train_file.write("{}\t{}\n".format(img_path, lidar_path))
            train_file.write("{}\n".format(img_path))

        for sample_index in range(first_test_sample_index, len(sample_img_pathes)):
            # Writing image and lidar paths for test samples
            img_path = sample_img_pathes[sample_index]
            filename = Path(img_path).stem
            lidar_path = str(Path(args.lidar_2_dir, (str(filename) + ".npy"))) # Assuming lidar has the same base name as the image
            #test_file.write("{}\t{}\n".format(img_path, lidar_path))
            test_file.write("{}\n".format(img_path))

if __name__ == "__main__":
    main()
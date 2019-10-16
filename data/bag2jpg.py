import sys
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
from fastai.vision import Path


if not os.path.exists("data"):
    os.makedirs("data")

path_main = Path("./bag_data")
for path_i in path_main.ls():
	print("Checking:",path_i)
	found=False
	while not found:
		try:
			path_i = path_i.ls()
			bags_c = []
			count=0
			for b in path_i:
				if os.path.splitext(str(b))[1] == ".bag":
				    bags_c.append(str(b))
				    count+=1
			if count!=0:
				found = True
		except:
			print("No bags files found")
			break

	print("{} bag files found.".format(count))

	pe = 0
	for index,b in enumerate(bags_c):
		dir_out = "../data/"+os.path.basename(path_i)+"/{}".format(index+pe)
		while os.path.exists(dir_out):
			pe+=1
    		dir_out = "../data/"+os.path.basename(path_i)+"/{}".format(index+pe)
		print(dir_out, b)
		if not os.path.exists(dir_out):
			os.makedirs(dir_out)
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_device_from_file(b, False)
		profile = pipeline.start(config)
		for _ in range(5):
			pipeline.wait_for_frames()
		colorizer = rs.colorizer()
		frame_id = 1
		while True:
			try:
				frames = pipeline.wait_for_frames()
				rgb_frame = frames.get_color_frame()
				if not rgb_frame:
					break
				color_image = np.asanyarray(rgb_frame.get_data())
				color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
				cv2.imwrite(os.path.join(dir_out, "rgb_frame_{:04d}.jpg".format(frame_id)), color_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
				frame_id += 1
			except RuntimeError:
				break
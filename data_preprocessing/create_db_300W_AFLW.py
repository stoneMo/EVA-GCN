import scipy.io as sio
# import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import os 
import sys
import cv2
import torch
# from moviepy.editor import *
import numpy as np
import argparse
import face_alignment   # FAN
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='../../data/AFLW2000',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='./AFLW2000.npz',
	                    help="path to output database mat file")
	parser.add_argument("--output_hm", type=str, default='./AFLW2000_hm.npz',
	                    help="path to output database mat file")
	parser.add_argument("--img_size", type=int, default=256,
	                    help="output image size")
	parser.add_argument("--ad", type=float, default=0.6,
	                    help="enlarge margin")


	args = parser.parse_args()
	return args

def gen_landmarks(img):
	key_points = fa.get_landmarks(img)
	if isinstance(key_points, list):
		if len(key_points) > 1:
			face_id = 0
			face_point = 0
			for i, points in enumerate(key_points):
				point = points[0][0]
				if point > face_point:
					face_point = point
					face_id = i
			key_point = key_points[face_id]
		else:
			key_point = key_points[0]
	
		if len(key_points) > 0:
			landmarks = key_point

	else:
		landmarks = None
	
	return landmarks

def visual_landmarks(image, landmarks):
	for i in range(len(landmarks)):
		x = landmarks[i][0]
		y = landmarks[i][1]
		image = cv2.circle(image, (x,y), radius=1, color=(0, 0, 255), thickness=1)
	
	return image

def gen_gaussian_heatmaps(img, landmarks, down_ratio, num_points=68):

	img_h, img_w = img.shape[:2]
	landmarks = landmarks / img_h

	map_height = img_h//down_ratio
	map_width = img_w//down_ratio
	heatmap=np.zeros((map_height, map_width, num_points),dtype=np.float)
	assert(len(landmarks)==num_points)
	for p in range(len(landmarks)):
		x=landmarks[p][0]*map_width
		y=landmarks[p][1]*map_height
		for i in range(map_width):
			for j in range(map_height):
				if (x-i)*(x-i)+(y-j)*(y-j)<=4:
					# print(1.0/(1+(x-i)*(x-i)*2+(y-j)*(y-j)*2))
					heatmap[j][i][p]=1.0/(1+(x-i)*(x-i)*2+(y-j)*(y-j)*2)

	return heatmap


def main():
	args = get_args()
	mypath = args.db
	output_path = args.output
	output_hm_path = args.output_hm

	img_size = args.img_size
	ad = args.ad

	isPlot = False

	onlyfiles_mat = []
	onlyfiles_jpg = []

	for root, dirs, files in os.walk(args.db, topdown=True):
		for name in files:
			file_name = os.path.join(root, name)
			# print("file_name:", file_name)
			if isfile(file_name) and file_name.endswith('.jpg'):
				onlyfiles_jpg.append(file_name)
			elif isfile(file_name) and file_name.endswith('.mat'):
				onlyfiles_mat.append(file_name)
	
	onlyfiles_mat.sort()
	onlyfiles_jpg.sort()
	print(len(onlyfiles_jpg))
	print(len(onlyfiles_mat))
	out_imgs = []
	out_poses = []
	# out_heatmaps = []
	out_landmarks = []

	for i in tqdm(range(len(onlyfiles_jpg))):
		img_name = onlyfiles_jpg[i]
		mat_name = onlyfiles_mat[i]

		img_name_split = img_name.split('.')
		mat_name_split = mat_name.split('.')

		if img_name_split[0] != mat_name_split[0]:
			print('Mismatched!')
			sys.exit()

		mat_contents = sio.loadmat(mat_name)
		# print("mat_name:", mat_name)
		if 'Pose_Para' not in list(mat_contents.keys()):
			continue
		else:
			pose_para = mat_contents['Pose_Para'][0]
		pt2d = mat_contents['pt2d']
		
		pt2d_x = pt2d[0,:]
		pt2d_y = pt2d[1,:]

		# I found negative value in AFLW2000. It need to be removed.
		pt2d_idx = pt2d_x>0.0
		pt2d_idy= pt2d_y>0.0

		pt2d_id = pt2d_idx
		if sum(pt2d_idx) > sum(pt2d_idy):
			pt2d_id = pt2d_idy
		
		pt2d_x = pt2d_x[pt2d_id]
		pt2d_y = pt2d_y[pt2d_id]
		
		img = cv2.imread(img_name)
		img_h = img.shape[0]
		img_w = img.shape[1]

		# Crop the face loosely
		x_min = int(min(pt2d_x))
		x_max = int(max(pt2d_x))
		y_min = int(min(pt2d_y))
		y_max = int(max(pt2d_y))
		
		h = y_max-y_min
		w = x_max-x_min

		# ad = 0.4
		x_min = max(int(x_min - ad * w), 0)
		x_max = min(int(x_max + ad * w), img_w - 1)
		y_min = max(int(y_min - ad * h), 0)
		y_max = min(int(y_max + ad * h), img_h - 1)
		
		img = img[y_min:y_max,x_min:x_max]
		# Checking the cropped image
		# if isPlot:
		# 	# cv2.imshow('check',img)
		# 	# k=cv2.waitKey(500)
		# 	cv2.imwrite(str(i)+'.jpg', img)
		
		# 	print("img:", img.shape)

		img = cv2.resize(img, (img_size, img_size))

		# to generate landmark and groundtruth heatmaps

		landmarks = gen_landmarks(img)

		if not isinstance(landmarks, np.ndarray):
			continue

		if isPlot:

			landmarks = gen_landmarks(img)

			print("landmarks:", landmarks.shape)

			down_ratio = 1     # (256, 256)
			heatmaps = gen_gaussian_heatmaps(img, landmarks, down_ratio)

			print("heatmap:", heatmaps.shape)

			# cv2.imshow('check',img)
			# k=cv2.waitKey(500)
			cv2.imwrite(str(i)+'.jpg', img)

			img_landmarks = visual_landmarks(img, landmarks)

			cv2.imwrite(str(i)+'_landmarks.jpg', img_landmarks)
			
			heatmap_img=np.zeros((256,256),dtype=np.float)
			for index in range(68):
				heatmap_img+=heatmaps[:,:,index]*255.0
			print(heatmap_img)

			Image.fromarray(heatmap_img).convert('RGB').save('{}_heatmaps.jpg'.format(i))

			print("img:", img.shape)
		
		pitch = pose_para[0] * 180 / np.pi
		yaw = pose_para[1] * 180 / np.pi
		roll = pose_para[2] * 180 / np.pi

		pose_labels = np.array([yaw, pitch, roll])

		out_imgs.append(img)
		out_poses.append(pose_labels)
		out_landmarks.append(landmarks)

	np.savez(output_path, image=np.array(out_imgs), landmark=np.array(out_landmarks), pose=np.array(out_poses), img_size=img_size)
	# np.savez(output_hm_path, heatmap=np.array(out_heatmaps))


if __name__ == '__main__':
	main()
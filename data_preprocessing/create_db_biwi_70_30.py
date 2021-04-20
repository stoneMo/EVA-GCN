import scipy.io as sio
# import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
import torch
# from moviepy.editor import *
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN
import face_alignment   # FAN
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)

def get_args():
	parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
	                                             "and creates database for training.",
	                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--db", type=str, default='../../data/BIWI/hpdb',
	                    help="path to database")
	parser.add_argument("--output", type=str, default='./BIWI',
	                    help="path to output database mat file")
	parser.add_argument("--img_size", type=int, default=256,
	                    help="output image size")
	parser.add_argument("--ad", type=float, default=0.4,
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
	img_size = args.img_size
	ad = args.ad

	isPlot = False
	detector = MTCNN()

	randFlag = np.zeros(24)
	randFlag[0:16] = 1
	randFlag = np.random.permutation(randFlag)

	print(randFlag)
	output_train_path = output_path+'_train.npz'
	output_test_path = output_path+'_test.npz'

	onlyfiles_png = []
	onlyfiles_txt = []
	for num in range(0,1):
		if num<9:
			mypath_obj = mypath+'/0'+str(num+1)
		else:
			mypath_obj = mypath+'/'+str(num+1)
		print(mypath_obj)
		onlyfiles_txt_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.txt')]
		onlyfiles_png_temp = [f for f in listdir(mypath_obj) if isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.png')]
	
		onlyfiles_txt_temp.sort()
		onlyfiles_png_temp.sort()

		onlyfiles_txt.append(onlyfiles_txt_temp)
		onlyfiles_png.append(onlyfiles_png_temp)
	print(len(onlyfiles_txt))
	print(len(onlyfiles_png))
	
	out_imgs_train = []
	out_poses_train = []
	out_landmarks_train = []

	out_imgs_test = []
	out_poses_test = []
	out_landmarks_test = []

	for i in range(len(onlyfiles_png))[:5]:
		print('object %d' %i)
		
		mypath_obj = ''
		if i<9:
			mypath_obj = mypath+'/0'+str(i+1)
		else:
			mypath_obj = mypath+'/'+str(i+1)

		for j in tqdm(range(len(onlyfiles_png[i]))):
			
			img_name = onlyfiles_png[i][j]
			txt_name = onlyfiles_txt[i][j]
			
			img_name_split = img_name.split('_')
			txt_name_split = txt_name.split('_')

			if img_name_split[1] != txt_name_split[1]:
				print('Mismatched!')
				sys.exit()


			pose_path = mypath_obj+'/'+txt_name
			# Load pose in degrees
			pose_annot = open(pose_path, 'r')
			R = []
			for line in pose_annot:
				line = line.strip('\n').split(' ')
				L = []
				if line[0] != '':
					for nb in line:
						if nb == '':
							continue
						L.append(float(nb))
					R.append(L)

			R = np.array(R)
			T = R[3,:]
			R = R[:3,:]
			pose_annot.close()

			R = np.transpose(R)

			roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
			yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
			pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi



			img = cv2.imread(mypath_obj+'/'+img_name)
			img_h = img.shape[0]
			img_w = img.shape[1]
			if j==0:
				[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [0,0,0,0]
			detected = detector.detect_faces(img)

			if len(detected) > 0:
				dis_list = []
				XY = []
				for i_d, d in enumerate(detected):
					
					xv = []
					yv = []
					print('keypoints', len(d['keypoints']))
					for key, value in d['keypoints'].items():
						xv.append(value[0]) 
						yv.append(value[1])
					
					if d['confidence'] > 0.95:
						x1,y1,w,h = d['box']
						x2 = x1 + w
						y2 = y1 + h
						xw1 = max(int(x1 - ad * w), 0)
						yw1 = max(int(y1 - ad * h), 0)
						xw2 = min(int(x2 + ad * w), img_w - 1)
						yw2 = min(int(y2 + ad * h), img_h - 1)
						
						# Crop the face loosely
						# x_min = int(min(xv))
						# x_max = int(max(xv))
						# y_min = int(min(yv))
						# y_max = int(max(yv))
						
						# h = y_max-y_min
						# w = x_max-x_min

						# xw1 = max(int(x_min - ad * w), 0)
						# xw2 = min(int(x_max + ad * w), img_w - 1)
						# yw1 = max(int(y_min - ad * h), 0)
						# yw2 = min(int(y_max + ad * h), img_h - 1)

						XY.append([xw1,xw2,yw1,yw2])
						dis_betw_cen = np.abs(xw1-img_w*2/3)+np.abs(yw1-img_h*2/3)
						dis_list.append(dis_betw_cen)
				
				if len(dis_list)>0:
					min_id = np.argmin(dis_list)
					[xw1,xw2,yw1,yw2] = XY[min_id]


				dis_betw_frames = np.abs(xw1-xw1_pre)
				if dis_betw_frames < 80 or j==0:
					img = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
					[xw1_pre,xw2_pre,yw1_pre,yw2_pre] = [xw1,xw2,yw1,yw2]
					if isPlot:
						print([xw1_pre,xw2_pre,yw1_pre,yw2_pre])
						# cv2.imshow('check',img)
						# k=cv2.waitKey(10)
					img = cv2.resize(img, (img_size, img_size))
					# # to generate landmark and groundtruth heatmaps
					landmarks = gen_landmarks(img)
					if not isinstance(landmarks, np.ndarray):
						continue
					# print("landmarks:", landmarks.shape)
					# down_ratio = 1     # (256, 256)
					# heatmaps = gen_gaussian_heatmaps(img, landmarks, down_ratio)
					# print("heatmap:", heatmaps.shape)

					cont_labels = np.array([yaw, pitch, roll])
					
					if randFlag[i] == 1:
						out_imgs_train.append(img)
						out_poses_train.append(cont_labels)
						out_landmarks_train.append(landmarks)
					elif randFlag[i] == 0:
						out_imgs_test.append(img)
						out_poses_test.append(cont_labels)
						out_landmarks_test.append(landmarks)
				else:
					img = cv2.resize(img[yw1_pre:yw2_pre + 1, xw1_pre:xw2_pre + 1, :], (img_size, img_size))
					# Checking the cropped image
					if isPlot:
						print([xw1_pre,xw2_pre,yw1_pre,yw2_pre])
						print('Distance between two frames too large! Use previous frame detected location.')
					
						# cv2.imshow('check',img)
						# k=cv2.waitKey(10)

					img = cv2.resize(img, (img_size, img_size))
					# # to generate landmark and groundtruth heatmaps
					landmarks = gen_landmarks(img)
					if not isinstance(landmarks, np.ndarray):
						continue
					# print("landmarks:", landmarks.shape)
					# down_ratio = 1     # (256, 256)
					# heatmaps = gen_gaussian_heatmaps(img, landmarks, down_ratio)
					# print("heatmap:", heatmaps.shape)

					cont_labels = np.array([yaw, pitch, roll])
					if randFlag[i] == 1:
						out_imgs_train.append(img)
						out_poses_train.append(cont_labels)
						out_landmarks_train.append(landmarks)
					elif randFlag[i] == 0:
						out_imgs_test.append(img)
						out_poses_test.append(cont_labels)
						out_landmarks_test.append(landmarks)
			else:
				img = cv2.resize(img[yw1_pre:yw2_pre + 1, xw1_pre:xw2_pre + 1, :], (img_size, img_size))
				if isPlot:
					print('No face detected! Use previous frame detected location.')
				
		
				img = cv2.resize(img, (img_size, img_size))

				# # to generate landmark and groundtruth heatmaps
				landmarks = gen_landmarks(img)
				if not isinstance(landmarks, np.ndarray):
					continue
				# print("landmarks:", landmarks.shape)
				# down_ratio = 1     # (256, 256)
				# heatmaps = gen_gaussian_heatmaps(img, landmarks, down_ratio)
				# print("heatmap:", heatmaps.shape)

				# Checking the cropped image
				if isPlot:
					# cv2.imshow('check',img)
					# k=cv2.waitKey(10)
					cv2.imwrite(str(j)+'.jpg', img)

					img_landmarks = visual_landmarks(img, landmarks)

					cv2.imwrite(str(j)+'_landmarks.jpg', img_landmarks)
					
					heatmap_img=np.zeros((256,256),dtype=np.float)
					for index in range(68):
						heatmap_img+=heatmaps[:,:,index]*255.0
					print(heatmap_img)

					Image.fromarray(heatmap_img).convert('RGB').save('{}_heatmaps.jpg'.format(j))
				
					print("img:", img.shape)


				cont_labels = np.array([yaw, pitch, roll])
				if randFlag[i] == 1:
					out_imgs_train.append(img)
					out_poses_train.append(cont_labels)
					out_landmarks_train.append(landmarks)
				elif randFlag[i] == 0:
					out_imgs_test.append(img)
					out_poses_test.append(cont_labels)
					out_landmarks_test.append(landmarks)

	np.savez(output_train_path,image=np.array(out_imgs_train), landmark=np.array(out_landmarks_train), pose=np.array(out_poses_train), img_size=img_size)
	np.savez(output_test_path,image=np.array(out_imgs_test), landmark=np.array(out_landmarks_test), pose=np.array(out_poses_test), img_size=img_size)


if __name__ == '__main__':
	main()
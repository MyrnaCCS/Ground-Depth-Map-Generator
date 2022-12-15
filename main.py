import cv2
import numpy as np
import os, re
from glob import glob
from GroundMapGenerator import GroundMapGenerator

dataset_path = '/home/myrna/Desktop/Indoors_Dataset/data'
dataset_path = '/home/jbhayet/opt/repositories/devel/Obstacle-detection-and-depth-estimation/data/INDOORS/'
#sequences_dir = ['40_D', '41_D', '42_D', '43_D', '44_D', '45_D', '46_D', '47_D', '48_D', '49_D', '50_D', '51_D', '52_D', '53_D', '54_D', '55_D', '56_D', '57_D']
sequences_dir = ['21_D','22_D','23_D','24_D','25_D','26_D','27_D','28_D','29_D','31_D','37_D']
#sequences_dir = ['20_D']
K = np.array([[616.4, 0., 319.], [0., 611.21, 239.], [0., 0., 1.]])
depth_map_generator = GroundMapGenerator(K, depth_max=7.0, width=640, height=480)


for seq in sequences_dir:
	transformation_matrix_dirs = sorted(glob(os.path.join(dataset_path, seq, 'transformation_matrix', '*' + '.txt')))

	ground_map_dir = os.path.join(dataset_path, seq, 'ground_plane_z')

	os.mkdir(ground_map_dir)

	for idx, matrix_path in enumerate(transformation_matrix_dirs):
		print(matrix_path)
		n   = re.findall('\d+', matrix_path )
		num = int(n[1])

		depth_map = depth_map_generator.compute_ground_plane_depth_map(matrix_path, rows=160, cols=256)
		cv2.imwrite(os.path.join(ground_map_dir, str(num).zfill(5) + '.png'), depth_map)

import numpy as np


class GroundMapGenerator(object):
	def __init__(self, K, depth_max, width, height):
		self.K = K #Intrisic Parameters
		self.width = width #Camera intrisic parameter
		self.height = height #Camera intrisic parameter
		self.depth_max = depth_max


	def read_transformation_matrix(self, transformation_matrix_path):
		file_rows = open(transformation_matrix_path).readlines()

		T = np.zeros((3, 4))

		for i in range(3):
			row = file_rows[i].split(' ')
			for j in range(4):
				T[i, j] = float(row[j])

		return T


	def compute_3D_coordinates(self, point_2D, P):
		A = np.zeros((2, 2))
		b = np.zeros((2, 1))

		# Rotation matrix dot K
		R = P[:, :3]
		# Translation vector dot K
		t = P[:, 3]

		# Equations system type: Ax = b
		A[0, 0] = (point_2D[0] * R[2, 0]) - R[0, 0]
		A[0, 1] = (point_2D[0] * R[2, 1]) - R[0, 1]
		A[1, 0] = (point_2D[1] * R[2, 0]) - R[1, 0]
		A[1, 1] = (point_2D[1] * R[2, 1]) - R[1, 1]
		b[0, 0] = t[0] - (point_2D[0] * t[2])
		b[1, 0] = t[1] - (point_2D[1] * t[2])

		# Solution: A-1b = x
		x = np.dot(np.linalg.inv(A), b)

		return np.insert(x,2,0.0,axis=0)


	def compute_depth_value(self, point_2D, T):
		# Projection matrix: from Wm to Wc
		P = np.dot(self.K, T)

		# Get 3D coordinates of points such that Zm = 0
		point_3D = self.compute_3D_coordinates(point_2D, P)

		# Get R and t: from Wm to Wc
		R = T[:, :3]
		t = T[:, 3]

		# Location of camera in Wm
		R_inv = np.transpose(R)
		cam   = np.dot(R_inv, -t)

		# Depth:
		depth_cam = np.sqrt((cam[0] - point_3D[0])**2 + (cam[1] - point_3D[1])**2 + (cam[2])**2)
		z_cam   = np.dot(R,point_3D-np.expand_dims(cam,-1))
		return depth_cam, z_cam[2]


	def compute_ground_plane_depth_map(self, transformation_matrix_path, rows, cols, compute_z=True):
		T = self.read_transformation_matrix(transformation_matrix_path)

		point_2D = np.ones((3, 1))

		# return depth map
		depth_map = np.ones((rows, cols)) * 255

		for i in range(cols):
			for j in range(rows):
				point_2D[0] = self.width/2.0 + (float(i-cols/2.0) / cols) * self.width
				point_2D[1] = self.height/2.0+ (float(j-rows/2.0) / rows) * self.height
				depth, z = self.compute_depth_value(point_2D, T)
				if compute_z==False:
					if depth <= self.depth_max:
						depth_map[rows-1-j, cols-1-i] = int(np.floor(depth / self.depth_max * 255.0))
					else:
						depth_map[rows-1-j, cols-1-i] = 255
						break
				else:
					if z <= self.depth_max:
						depth_map[rows-1-j, cols-1-i] = int(np.floor(z / self.depth_max * 255.0))
					else:
						depth_map[rows-1-j, cols-1-i] = 255
						break

		return depth_map

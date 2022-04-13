import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy import spatial
from scipy.ndimage.morphology import binary_fill_holes


def prune(inds):
	# PRUNING STEP. remove branches with fewer than 25 points
	n4neigh = [spatial.cKDTree(inds).query(pt, k=4, p=2, distance_upper_bound=1.5) for pt in inds]
	n4dist = [row[0] for row in n4neigh]
	n4pts = [row[1] for row in n4neigh]

	# find end points
	end_pts = []
	for i in range(len(n4dist)):		
		valids = n4dist[i][np.where(n4dist[i] < 1.5)]
		if len(valids) == 2: 	# an endpoint
			end = n4pts[i][np.where(n4dist[i] == 0)]
			end_pts.append(end[0])

	# find all the coordinates from an end point to the 
	# nearest bifurcation point
	branches = {}
	pts_to_remove = np.empty(0)

	for pt in end_pts:
		branch = np.empty(0)
		prev_pt, curr_pt = pt, pt
		while True:
			valids = n4pts[curr_pt][np.where(n4dist[curr_pt] < 1.5)[0]]
			if len(valids) > 3: 	# bifurcation point
				break
			if len(valids) == 2 and curr_pt != pt: 	# another end point
				break
			branch = np.append(branch, curr_pt)
			next_pt = valids[np.where((valids != curr_pt) & (valids != prev_pt))[0]][0]
			prev_pt = curr_pt
			curr_pt = next_pt
		if len(branch) > 5: 	# skip main branch !!! choose an appropriate number
			continue
		pts_to_remove = np.append(pts_to_remove, branch)

	inds = np.delete(inds, pts_to_remove.astype(np.int), 0)

	# RE-SKELETONIZE. need to remove one-pixel branches
	pruned = np.zeros((128, 128))
	# pruned = np.zeros((512, 512))
	for pt in range(inds.shape[0]):
		pruned[inds[pt, 0], inds[pt, 1]] = 255
	pruned = skeletonize(pruned > 0, method='lee')

	inds = np.where(pruned > 0)
	ind_row = np.expand_dims(inds[0], axis=1)
	ind_col = np.expand_dims(inds[1], axis=1)
	inds = np.concatenate((ind_row, ind_col), axis=1)

	return inds


class MainCurve(object):
	# perform mask operations
	def __init__(self, mask, op_name):
		mask[mask < 20] = 0
		_, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
		largest_label = np.where(stats[1:, 4] == max(stats[1:, 4]))[0] 	# zeroth label is background
		mask = labeled == (largest_label+1) 	# zeroth label here is actually first label
		mask = binary_fill_holes(mask)
		mask = mask.astype('uint8') * 255

		inds = np.where(mask > 0)
		self.inds_mask = [tuple((inds[0][index], inds[1][index])) for index in range(inds[0].shape[0])]

		# find the center line
		mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
		mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
		skeleton = skeletonize(mask > 0)

		self.img_dimensions = skeleton.shape

		inds = np.where(skeleton > 0)
		ind_row = np.expand_dims(inds[0], axis=1)
		ind_col = np.expand_dims(inds[1], axis=1)
		inds = np.concatenate((ind_row, ind_col), axis=1)

		self.inds_first_skel = inds

		bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		for p_ind in range(len(self.inds_first_skel)):
			bgr_mask[self.inds_first_skel[p_ind][0], self.inds_first_skel[p_ind][1], :] = [255, 0, 0]

		# PRUNING
		prune_again, loop = True, 0
		while loop < 100 and prune_again:
			prune_again = False
			loop += 1
			# print(inds)
			inds = prune(inds)
			# print(inds)
			n4neigh = [spatial.cKDTree(inds).query(pt, k=4, p=2, distance_upper_bound=1.5) for pt in inds]
			n4dist = [row[0] for row in n4neigh]

			# print(loop)
			for i in range(len(n4dist)):
				valids = n4dist[i][np.where(n4dist[i] < 1.5)]
				# check if there are any one-point branch that may mess up ordering
				if len(valids) == 4:
					prune_again = True

		self.inds_pruned = inds

		if loop >= 100:
			for p_ind in range(len(self.inds_pruned)):
				bgr_mask[self.inds_pruned[p_ind][0], self.inds_pruned[p_ind][1], :] = [0, 255, 0]
			cv2.imwrite(op_name, bgr_mask)
			sys.exit("Warning: Looped 100 times, remaining branches may mess up ordering.")

		# CHECKPOINT. find starting point of center line
		n4neigh = [spatial.cKDTree(inds).query(pt, k=4, p=2, distance_upper_bound=1.5) for pt in inds]
		n4dist = [row[0] for row in n4neigh]

		start_pt_found_flag = False
		end_pt_found_flag = False
		for i in range(len(n4dist)):
			valids = n4dist[i][np.where(n4dist[i] < 1.5)]
			# print(len(valids))
			if len(valids) == 2:
				if not start_pt_found_flag:
					start_pt = i
					start_pt_found_flag = True
				elif not end_pt_found_flag:
					end_pt = i
					end_pt_found_flag = True
				else:
					sys.exit("Warning: More than two end points.")
			if len(valids) == 4:
				print(valids)
				sys.exit("Warning: Remaining branches may mess up ordering.")

		# print(start_pt_found_flag, end_pt_found_flag, n4dist, self.inds_first_skel.shape)
		if inds[start_pt][0] > inds[end_pt][0]:
			start_pt, end_pt = end_pt, start_pt
		if inds[start_pt][1] > inds[end_pt][1]:
			with open('/home/chentyt/Documents/4tb/Tiana/P100/Data/RCA_annotated_v2/Warning.log', 'a') as log:
				log.write('{} Start point {} End point {} \n'.format(op_name, inds[start_pt], inds[end_pt]))
			# print('Unclear start point and end point.')

		# print('Start point:', inds[start_pt], 'End point:', inds[end_pt])

		# ORDERING STEP. need to get correct sequence of points to generate deformation control points.
		nearest_neighbors = [spatial.cKDTree(inds).query(pt, k=3) for pt in inds]
		# print(nearest_neighbors)
		distances = [0]
		seq = []

		# find the next point on the center line
		neighbors = list(nearest_neighbors[start_pt][1])
		# print(nearest_neighbors[start_pt])
		# print(neighbors)
		neighbors.remove(start_pt)
		neighbors.remove(nearest_neighbors[start_pt][1][np.where(nearest_neighbors[start_pt][0] > 1.5)[0][0]])
		# print(neighbors, start_pt, nearest_neighbors[start_pt][1][np.where(nearest_neighbors[start_pt][0] > 1.5)[0][0]])

		# prev_pt = start_pt
		curr_pt = neighbors[0]
		seq.append(start_pt)
		seq.append(curr_pt)

		for i in range(inds.shape[0]-2):
			neighbors = list(nearest_neighbors[curr_pt][1])
			# convert list to an int by indexing into [0].

			# next_pt = [pt for pt in neighbors if pt not in (curr_pt, prev_pt)][0]
			next_pt = [pt for pt in neighbors if pt not in seq][0]		# can prevent duplication

			# distances.append(np.sqrt((inds[curr_pt, 0]-inds[next_pt, 0])**2 + (inds[curr_pt, 1]-inds[next_pt, 1])**2))
			dis = np.linalg.norm(inds[curr_pt] - inds[next_pt])
			# print(next_pt, dis)
			seq.append(next_pt)
			distances.append(dis)
		
			# prev_pt = curr_pt
			curr_pt = next_pt

		if len(seq) != inds.shape[0]:
			sys.exit("Warning: Flaw in ordering")

		self.inds_ordered = [inds[i] for i in seq]

		# bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		# # print(bgr_mask.shape)
		#
		# for p_ind in range(len(self.inds_first_skel)):
		# 	bgr_mask[self.inds_first_skel[p_ind][0], self.inds_first_skel[p_ind][1], 0] = 255
		# 	bgr_mask[self.inds_first_skel[p_ind][0], self.inds_first_skel[p_ind][1], 1] = 0
		# 	bgr_mask[self.inds_first_skel[p_ind][0], self.inds_first_skel[p_ind][1], 2] = 0

		for p_ind in range(len(self.inds_ordered)):
			bgr_mask[self.inds_ordered[p_ind][0], self.inds_ordered[p_ind][1], :] = [0, 255, 0]

		bgr_mask[self.inds_ordered[0][0], self.inds_ordered[0][1], :] = [0, 0, 255]
		# cv2.imwrite(op_name, bgr_mask)

		self.bgr_mask = bgr_mask

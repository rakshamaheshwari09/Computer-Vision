import cv2
import numpy as np
import inspect, sys, re, operator
from model import Trainer
from solver import Solver
import torch

class Detector:
	def __init__(self):
		p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

		self.stages = list(sorted(
		map(
			lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
			filter(
				lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
				inspect.getmembers(self))),
		key=lambda x: x[0]))

		# For storing the recognized digits
		self.digits = [ [None for i in range(9)] for j in range(9) ]

	# Takes as input 9x9 array of numpy images
	# Combines them into 1 image and returns
	# All 9x9 images need to be of same shape
	def makePreview(images):
		assert isinstance(images, list)
		assert len(images) > 0
		assert isinstance(images[0], list)
		assert len(images[0]) > 0
		assert isinstance(images[0], list)

		rows = len(images)
		cols = len(images[0])

		cellShape = images[0][0].shape

		padding = 10
		shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)
		
		result = np.full(shape, 255, np.uint8)

		for row in range(rows):
			for col in range(cols):
				pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

				result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

		return result


	# Takes as input 9x9 array of digits
	# Prints it out on the console in the form of sudoku
	# None instead of number means that its an empty cell
	def showSudoku(array):
		cnt = 0
		for row in array:
			if cnt % 3 == 0:
				print('+-------+-------+-------+')

			colcnt = 0
			for cell in row:
				if colcnt % 3 == 0:
					print('| ', end='')
				print('. ' if cell is None else str(cell) + ' ', end='')
				colcnt += 1
			print('|')
			cnt += 1
		print('+-------+-------+-------+')

	# Runs the detector on the image at path, and returns the 9x9 solved digits
	# if show=True, then the stage results are shown on screen
	# Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
	# that the digit at (1,2) is corrected to 9
	# and the digit at (3,3) is corrected to 4
	def run(self, path='assets/sudokus/sudoku2.jpg', show = False, corrections = []):
		self.path = path
		self.original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

		self.run_stages(show)
		result = self.solve(corrections)


		if show:
			self.showSolved()
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return result

	# Runs all the stages
	def run_stages(self, show):
		results = [('Original', self.original)]

		for idx, name, fun in self.stages:
			image = fun().copy()
			results.append((name, image))

		if show:
			for name, img in results:
				cv2.imshow(name, img)
		

	# Stages
	# Stage function name format: stage_[stage index]_[stage name]
	# Stages are executed increasing order of stage index
	# The function should return a numpy image, which is shown onto the screen
	# In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
	# to create a single image out of those
	# You can pass data from one stage to another using class member variables
	def stage_1_preprocess(self):
		image = self.original.copy()
		image = cv2.GaussianBlur(image, (9,9), 0)
		image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		image = cv2.bitwise_not(image, image)  


		kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
		image = cv2.dilate(image, kernel)

		self.image1 = image
		return self.image1

	def stage_2_resize(self):
		
		image = cv2.resize(self.image1, (28, 28))

		cells = [[image.copy() for i in range(9)] for j in range(9)]

		return Detector.makePreview(cells)

	def stage_3_poly(self):
		contours, _ = cv2.findContours(self.image1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		polygon = contours[0]

		bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
		points = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
		
		for i in range(4):
			cv2.circle(self.image1, (points[i][0], points[i][1]), 10, (0,0,255), cv2.FILLED)

		src = np.array([points[0], points[1], points[2], points[3]], dtype = np.float32) 
		print(type(src))

		def distance_between(list1=[0,0], list2=[0,0]):
			a = list2[0] - list1[0]
			b = list2[1] - list1[1]

			return np.sqrt((a**2) + (b**2))

		side = max([  distance_between(points[2], points[1]), 
            distance_between(points[0], points[3]),
            distance_between(points[2], points[3]),   
            distance_between(points[0], points[1]) ])

		dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype = np.float32)

		m = cv2.getPerspectiveTransform(src, dst)
		n = cv2.warpPerspective(self.original.copy(), m, (int(side), int(side)))
		self.image1 = n

		return self.image1

	def stage_4_digits(self):
		def infer_grid(img):
			"""Infers 81 cell grid from a square image."""
			squares = []
			side = img.shape[:1]
			side = side[0] / 9
			
			# Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
			for j in range(9):
				for i in range(9):
					p1 = (i * side, j * side)  # Top left corner of a bounding box
					p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
					squares.append((p1, p2))

			return squares

		def cut_from_rect(img, rect):
			"""Cuts a rectangle from an image using the top left and bottom right points."""
			return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

		def scale_and_centre(img, size, margin=0, background=0):
			"""Scales and centres an image onto a new background square."""
			h, w = img.shape[:2]
			
			def centre_pad(length):
				"""Handles centering for a given length that may be odd or even."""
				if length % 2 == 0:
					side1 = int((size - length) / 2)
					side2 = side1
				else:
					side1 = int((size - length) / 2)
					side2 = side1 + 1
				return side1, side2
					
			def scale(r, x):
				return int(r * x)
			
			if h > w:
				t_pad = int(margin / 2)
				b_pad = t_pad
				ratio = (size - margin) / h
				w, h = scale(ratio, w), scale(ratio, h)
				l_pad, r_pad = centre_pad(w)
			else:
				l_pad = int(margin / 2)
				r_pad = l_pad
				ratio = (size - margin) / w
				w, h = scale(ratio, w), scale(ratio, h)
				t_pad, b_pad = centre_pad(h)
				
			img = cv2.resize(img, (w, h))
			img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
			return cv2.resize(img, (size, size))
			
		def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
			img = inp_img.copy()  # Copy the image, leaving the original untouched
			height, width = img.shape[:2]
			
			max_area = 0
			seed_point = (None, None)
			
			if scan_tl is None:
				scan_tl = [0, 0]
				
			if scan_br is None:
				scan_br = [width, height]
				
			#Loop through the image
			 
			for x in range(scan_tl[0], scan_br[0]):
				for y in range(scan_tl[1], scan_br[1]):
					if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
						area = cv2.floodFill(img, None, (x, y), 64)
						if area[0] > max_area:  # Gets the maximum bound area which should be the grid
							max_area = area[0]
							seed_point = (x, y)
							
			for x in range(width):
				for y in range(height):
					if img.item(y, x) == 255 and x < width and y < height:
						cv2.floodFill(img, None, (x, y), 64)
						
			mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image
			
			if all([p is not None for p in seed_point]):
				cv2.floodFill(img, mask, seed_point, 255)
				
			top, bottom, left, right = height, 0, width, 0
			
			for x in range(width):
				for y in range(height):
					if img.item(y, x) == 64:  # Hide anything that isn't the main feature
						cv2.floodFill(img, mask, (x, y), 0)
						
					if img.item(y, x) == 255:
						top = y if y < top else top
						bottom = y if y > bottom else bottom
						left = x if x < left else left
						right = x if x > right else right
						
			bbox = [[left, top], [right, bottom]]
			return img, np.array(bbox, dtype='float32'), seed_point
		
		def extract_digit(img, rect, size):
			"""Extracts a digit (if one exists) from a Sudoku square."""
			digit = cut_from_rect(img, rect)  # Get the digit box from the whole square
			h, w = digit.shape[:2]
			margin = int(np.mean([h, w]) / 2.5)
			_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
			digit = cut_from_rect(digit, bbox)
			
			w = bbox[1][0] - bbox[0][0]
			h = bbox[1][1] - bbox[0][1]
			
			if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
				return scale_and_centre(digit, size, 4)
			else:
				return np.zeros((size, size), np.uint8)
		
		def get_digits(img, squares, size):
			"""Extracts digits from their cells and builds an array"""
			digits = []
			img = cv2.GaussianBlur(img, (9,9), 0)
			img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
			img = cv2.bitwise_not(img, img)  
			cv2.imshow('img', img)
			for square in squares:
				digits.append(extract_digit(img, square, size))
			return digits

		squares = infer_grid(self.image1)
		digits = get_digits(self.image1, squares, 28)
		self.digit = digits
		print(np.shape(digits[0]))

		def show_digits(digits, colour=255):
			"""Shows list of 81 extracted digits in a grid format"""
			rows = []
			with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
			for i in range(9):
				row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
				rows.append(row)

			img = np.concatenate(rows)
			return img

		final = show_digits(digits)
		return final

	def stage_5_recognize(self):
		function = Trainer()
		numbers = []
		for i in range(0,81):
			numbers.append(function.predict(self.digit[i]).numpy())
		print(numbers)

		return self.image1





	# Solve function
	# Returns solution
	def solve(self, corrections):
		# Only upto 3 corrections allowed
		assert len(corrections) < 3

		# Apply the corrections

		# Solve the sudoku
		self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
		s = Solver(self.answers)
		if s.solve():
			self.answers = s.digits
			return s.digits

		return [[None for i in range(9)] for j in range(9)]

	# Optional
	# Use this function to backproject the solved digits onto the original image
	# Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
	# an appropriate name
	def showSolved(self):
		pass


if __name__ == '__main__':
	d = Detector()
	result = d.run('assets/sudokus/sudoku2.jpg', show=True)
	print('Recognized Sudoku:')
	Detector.showSudoku(d.digits)
	print('\n\nSolved Sudoku:')
	Detector.showSudoku(result)
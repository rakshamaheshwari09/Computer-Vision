from mnist import MNIST
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import glob
import random

class PrintedMNIST(Dataset):
    """Generates images containing a single digit from font"""

    def __init__(self, N, random_state, transform=None):
        """"""
        self.N = N
        self.random_state = random_state
        self.transform = transform

        fonts_folder = "fonts"

        self.fonts = ["Roboto-Bold.ttf", "Rubik-Bold.ttf", "Raleway-Bold.ttf"]
        #self.fonts = glob.glob(fonts_folder + "/*.ttf")

        random.seed(random_state)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        target = random.randint(0, 9)

        size = random.randint(150, 250)
        x = random.randint(30, 90)
        y = random.randint(30, 90)

        color = random.randint(200, 255)

        # Generate image
        img = Image.new("L", (256, 256))

        target = random.randint(0, 9)

        size = random.randint(150, 250)
        x = random.randint(30, 90)
        y = random.randint(30, 90)

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(random.choice(self.fonts), size)
        draw.text((x, y), str(target), color, font=font)

        img = img.resize((28, 28), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)
			
        img = np.array(img)

        return img, target

class Loader:
	def __init__(self):

		# Instead of using MNIST(), you can generate your own dataset
		# of printed digits on the fly using the PIL library to generate image of digits

		data = PrintedMNIST(N=600, random_state=10)
		self.train_data = []
		self.train_labels = []
		for i in range (data.N-10000):
			print(i)
			d = data.__getitem__(10)
			#print(np.shape(np.array(d[0]).flatten()))
			self.train_data.append(np.array(d[0]).flatten())
			self.train_labels.append(d[1])

		self.test_data = []
		self.test_labels = []
		for i in range (600):
			print(i)
			d = data.__getitem__(100)
			self.test_data.append(np.array(d[0]).flatten())
			self.test_labels.append(d[1])

		self.train_data = np.array(self.train_data)
		self.train_labels = np.array(self.train_labels)
		self.test_data = np.array(self.test_data)
		self.test_labels = np.array(self.test_labels)

		self.train_data = np.array(self.train_data).astype(np.uint8)
		self.test_data = np.array(self.test_data).astype(np.uint8)

		self.train_labels = np.array(self.train_labels).astype(np.int32)
		self.test_labels = np.array(self.test_labels).astype(np.int32)


	# Shows a preview of 16 randomly selected images from the train dataset along with the labels
	def preview(self):
		indices = np.random.randint(0, self.test_data.shape[0], (4,4))
		images = [ [ self.test_data[indices[i,j]].reshape((28,28)) for j in range(4)] for i in range(4) ]

		final = np.full((28 * 4 + 4 * 20, 28 * 4 + 3 * 5),255,np.uint8)
		for i in range(4):
			for j in range(4):
				final[j*(28+20):j*(28+20)+28,i*(28+5):i*(28+5)+28] = images[i][j]
				final = cv2.putText(final,
					str(self.test_labels[indices[i,j]]),
					(i*(28+5), (j+1)*(28+18)),
					cv2.FONT_HERSHEY_COMPLEX,
					0.5,
					(0,0,0),
					1,
					cv2.LINE_AA)
		
		final = cv2.resize(final, (0,0), fx=2, fy=2)
		cv2.imshow('Preview', final)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	l = Loader()
	l.preview()
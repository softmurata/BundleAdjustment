import torch.nn as nn
import torch

class Tatarchenko16(nn.Module):
	def __init__(self):
		super(Tatarchenko16, self).__init__()

		self.conv0 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
		self.conv1 = nn.Conv2d(32, 32, 5, stride=2, padding=2)
		self.conv2 = nn.Conv2d(32,64,3,stride=2, padding=1)
		self.conv3 = nn.Conv2d(64,128,3,stride=2, padding=1)
		self.conv4 = nn.Conv2d(128,256,3,stride=2, padding=1)
		self.fci = nn.Linear(256*4*4, 1024)

		self.theta1 = nn.Linear(4,64)
		self.theta2 = nn.Linear(64,64)
		self.theta3 = nn.Linear(64,64)

		self.fc1 = nn.Linear(1024+64,1024)
		self.fc2 = nn.Linear(1024,1024)
		self.fc3 = nn.Linear(1024,256*4*4)

		self.dec4 = nn.ConvTranspose2d(256,128,3,stride=2, output_padding=1, padding=1)
		self.dec3 = nn.ConvTranspose2d(128,64,3,stride=2, output_padding=1, padding=1)
		self.dec2 = nn.ConvTranspose2d(64,32,5,stride=2, output_padding=1, padding=2)
		self.dec1 = nn.ConvTranspose2d(32,32,5,stride=2, output_padding=1, padding=2)
		self.dec0 = nn.ConvTranspose2d(32,4,5,stride=2, output_padding=1, padding=2)

		self.relu = nn.LeakyReLU(0.2)

	def forward(self, img, theta):
		img = self.relu(self.conv0(img))
		img = self.relu(self.conv1(img))
		img = self.relu(self.conv2(img))
		img = self.relu(self.conv3(img))
		img = self.relu(self.conv4(img))
		img = img.view(img.size(0), -1).contiguous()
		img = self.relu(self.fci(img))

		theta = self.relu(self.theta1(theta))
		theta = self.relu(self.theta2(theta))
		theta = self.relu(self.theta3(theta))


		fc = torch.cat((img, theta), 1)
		fc = self.fc1(fc)
		fc = self.fc2(fc)
		fc = self.fc3(fc)
		fc = fc.view(img.size(0), 256, 4, 4)
		img = self.relu(self.dec4(fc))
		img = self.relu(self.dec3(img))
		img = self.relu(self.dec2(img))
		img = self.relu(self.dec1(img))		
		img = (self.dec0(img))

		depth = img[:,3:,:,:]
		img = nn.Tanh()(img[:,0:3,:,:])

		return torch.cat((img, depth), 1)



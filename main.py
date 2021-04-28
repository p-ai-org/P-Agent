import airsim
import numpy as np
from torchvision import transforms
import torch
import PIL
import os
import io

from Network.HardCode_Controller import HardCode_Controller

# Check if Connection with Airsim is Good
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load CNN Model
loaded_model = torch.load(os.path.join(os.path.abspath(__name__),"..","Network","object_detection","mymodel3.pt"), map_location="cuda:0")		# Assumes model is stored in \P-agent\Network\Object_detection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Takeoff and Prepare for instructions
client.takeoffAsync().join()

done = False
mem = 0
while(done == False):
	# Grab image and convert to RGB tensor
	img = client.simGetImage("0", airsim.ImageType.Scene)
	f_im = PIL.Image.open(io.BytesIO(img)).convert('RGB')
	f_tensor = transforms.ToTensor()(f_im)

	# Feed image into CNN and collect prediction
	mask_pred = loaded_model([f_tensor.to(device)])
	if len(mask_pred[0]['boxes']) == 0:		# If empty tensor (ie no package)
		box_pred = []
		pass
	else:
		box_pred = mask_pred[0]['boxes'][0]         # First tensor is the bounding box coordinates
		confidence = mask_pred[0]['scores'][0]
		x0, y0, x1, y1 = [int(i) for i in box_pred.tolist()]		# Turn into coordinates
		box_pred = np.array([[x0,y0],[x1,y1]])			# First set of coordinates is top left of the box and second set is the bottom right
		
		if confidence == mask_pred[0]['scores'][0] < 0.9:		# Confidence threshold check
			box_pred = []

	Controller = HardCode_Controller(client, box_pred, mem)
	done, mem, collided = Controller.policy()

if collided:
	print("I hit a wall, dumdum")
if not collided:
	print("I got to the Package!")
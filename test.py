import cv2, os
from cv2 import dnn_superres

dir = os.path.dirname(os.path.abspath(__file__)) # get path of script.

print(dir)

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = f'{dir}\\ESPCN_x4.pb'
sr.readModel(path)

sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Read image
image = cv2.imread(f'{dir}\\input.png')

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 4)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite(f'{dir}\\output.png', result)
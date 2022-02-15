"""
Client For Testing
"""
import json
import requests
import numpy as np
from PIL import Image


img = Image.open("./images/4.jpeg")
arr = np.asarray(img)

url = f"http://127.0.0.1:5000/image/"
senData = {'arr': arr.tolist(), 'thr': 0.90}

response = requests.post(url, json=senData)
jsonData = response.json()
im = np.array(jsonData['image'], dtype=np.uint8)
labs = jsonData['labels']
resImage = Image.fromarray(im)
resImage.show()

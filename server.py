from flask import Flask, request, Response
import pickle
import numpy as np
from PIL import Image
import detect as dt


app = Flask(__name__)


@app.route('/')
def home():
    return "HOME"


@app.route("/image/", methods=["POST"])
def image():
    data = request.json
    img = np.array(data['arr'], dtype=np.uint8)
    thresh = data['thr'] or 0.92
    print(img.shape)
    img = Image.fromarray(img)
    resultImage, labels = dt.main(img, float(thresh))
    return {
        "image": np.asarray(resultImage).tolist(),
        "labels": labels
    }


if __name__ == "__main__":
    app.run(debug=True)

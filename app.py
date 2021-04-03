from flask import Flask, jsonify, request
import time
import base64

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms


app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.eval()


@app.route('/')
def index():
    return 'OK!'


@app.route("/test", methods=['POST'])
def response():
    query = dict(request.form)['query']
    result = query + " " + time.ctime()
    return jsonify({"response": result})


@app.route("/fruit_detection", methods=['POST'])
def predict():
    string = dict(request.form)['image']
    #  result = 'OK'
    jpg_original = base64.b64decode(string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(im_rgb)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    result_string = str()

    for i in range(top5_prob.size(0)):
        result_string += str(categories[top5_catid[i]]) + ' ' + str(top5_prob[i].item()) + '\n'

    return jsonify({"response": result_string})


if __name__ == "__main__":
    app.run()

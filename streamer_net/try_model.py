import cv2
import os

import numpy as np
import torch
from streamer_net.network_definition import SimpleCNN
from torchvision.transforms import ToTensor
transform = ToTensor()

vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)

print(os.listdir(os.getcwd()))

with open('emote_training_data/categories.txt') as f:
    categories = f.read().splitlines()

model = SimpleCNN(len(categories))
model.load_state_dict(torch.load("model"))


while (True):

    # inside infinity loop
    rect, frame = vid.read()
    cv2.imshow('frame', frame)
    prepared_frame = transform(frame)
    logits = model(prepared_frame)
    pred_probability = torch.nn.Softmax(dim=1)(logits)
    print(f"predicted class: {categories[pred_probability.argmax(1)[0]]}")
    print(f"confidence: {pred_probability[0][pred_probability.argmax(1)][0]}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
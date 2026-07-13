import cv2
import os

vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)
i=0

# inside infinity loop
rect, frame = vid.read()

cv2.imwrite(f"StreamerNet/streamer_net/emote_training_data/0/0.png", frame)

vid.release()
cv2.destroyAllWindows()
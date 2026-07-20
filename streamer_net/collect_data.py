import cv2
import os

vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)
i=0

while (True):

    # inside infinity loop
    rect, frame = vid.read()

    cv2.imwrite(f"streamer_net/emote_training_data/1/{i}.png", frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i+=1

vid.release()
cv2.destroyAllWindows()
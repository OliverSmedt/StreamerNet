import cv2
import os

vid = cv2.VideoCapture(0)
vid.set(3,200)
vid.set(4,200)
with open('emote_training_data/categories.txt') as f:
    categories = f.read().splitlines()

for i, category in enumerate(categories):
    j=0
    print(f"current category: {category}")
    if i!=len(categories)-1:
        print(f"next category: {categories[i+1]}")
    while True:
        rect, frame = vid.read()

        if not os.path.exists('emote_training_data/'+str(i)):
            os.makedirs('emote_training_data/'+str(i))

        cv2.imwrite("emote_training_data/{0}/{1}.png".format(i, j), frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        j+=1

vid.release()
cv2.destroyAllWindows()
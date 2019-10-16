import sys
import os
import cv2
import time
from PIL import Image
from nltk import edit_distance


def get_campaign(campaigns, word):
  return min(campaigns, key=lambda x: edit_distance(word, x))

sets = (
        "testing",
        "training",
)

loc = input("\nTraining or Testing: ")
word = input("\nWhich word: ")
directory = "../data/{set}_set/{word}".format(set=get_campaign(sets,loc), word=word)

imagecount = int(input("How many images: "))

os.makedirs(directory, exist_ok=True)

video = cv2.VideoCapture(0)

filename = len(os.listdir(directory))
count = 0

while True and count < imagecount:
        filename += 1
        count += 1
        _, frame = video.read()
        im = Image.fromarray(frame, 'RGB')
        im = im.resize((128,128))
        im.save(os.path.join(directory, str(filename)+".jpg"), "JPEG")

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
        time.sleep(0.2)
video.release()
cv2.destroyAllWindows()
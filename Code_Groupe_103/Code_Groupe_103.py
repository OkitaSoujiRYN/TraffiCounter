import cv2
import numpy as np
from time import sleep
import turtle
from turtle import *

# Longueur minimale du rectangle
min_length = 80
# Hauteur minimale du rectangle
min_height = 80
# Erreur tolérée entre les pixels
offset = 6
# Position de la ligne de comptage
pos_count = 550
# FPS de la vidéo
delay = 60

detect = []
cars = 0


def centering(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


turtle.setup(1280, 720)
turtle.bgpic("Fond.png")
ht()
up()
goto(-325, -15)
down()

turtle.color("orange")
turtle.clear()
write("Voulez-vous utiliser une caméra ou ouvrir une vidéo", font=("Comic sans ms", 20, ""))
choice = textinput("Choix", "Choisir votre option\n- caméra\n- vidéo\n")
while(choice != "caméra" and choice != "camera" and choice != "vidéo" and choice != "video"):
    turtle.clear()
    write("Voulez-vous utiliser une caméra ou ouvrir une vidéo", font=("Comic sans ms", 20, ""))
    choice = textinput("Choix", "Choisir votre option\n- caméra\n- vidéo\n")

if(choice == "camera" or choice == "caméra"):
    cap = cv2.VideoCapture(0)
    subtract = cv2.createBackgroundSubtractorKNN()
    pos_count = 400
else:
    turtle.clear()
    write("Saisir le nom de la vidéo", font=("Comic sans ms", 20, ""))
    choice = textinput("Nom", "Ecrire le nom de la vidéo")
    choice += ".mp4"
    cap = cv2.VideoCapture(choice)
    subtract = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtract.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (choice == "camera" or choice == "caméra"):
        cv2.line(frame1, (4, pos_count), (636, pos_count), (255, 127, 0), 3)
    else:
        cv2.line(frame1, (5, pos_count), (1275, pos_count), (255, 127, 0), 3)
    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        valid_contour = (w >= min_length) and (h >= min_height)
        if not valid_contour:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = centering(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (pos_count + offset) > y > (pos_count - offset):
                cars += 1
                if (choice == "camera" or choice == "caméra"):
                    cv2.line(frame1, (4, pos_count), (636, pos_count), (0, 127, 255), 3)
                else:
                    cv2.line(frame1, (5, pos_count), (1275, pos_count), (0, 127, 255), 3)
                detect.remove((x, y))
                turtle.clear()
                write("Car is detected : " + str(cars), font=("Comic sans ms", 20, ""))
    if(choice == "camera" or choice == "caméra"):
        cv2.putText(frame1, str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    else:
        cv2.putText(frame1, "VEHICLE COUNT : " + str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    cv2.imshow("Detect", dilated)
    cv2.imshow("Video Original", frame1)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

import sys
import pygame

from neural_network import *
import numpy as np
import pickle as pk

CACHE = r"C:\Users\marcu\OneDrive\Desktop\Python Scripts\Bigger Projects\AI\neural_network" + "\\" + input("What model would you like to interact with?: ")

with open(CACHE,"rb") as f:
    model : Neural_Network = pk.load(f)

size = 20
width = size*28
height = size*28

screen = pygame.display.set_mode((width,height))

image = [[0]*28 for i in range(28)]

pre = -1

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    left, middle, right = pygame.mouse.get_pressed()

    if right:
        image = [[0]*28 for i in range(28)]

    if left:
        mx, my = pygame.mouse.get_pos()

        x = mx // size
        y = my // size

        for i in range(-1,2):
            for j in range(-1,2):
                cx = x + i
                cy = y + j

                if 0 <= cx < 28 and 0 <= cy < 28:
                    px = (cx + 0.5) * size
                    py = (cy + 0.5) * size

                    add = 10 / (1 + ((px - mx)**2 + (py - my)**2)**0.5)

                    image[cy][cx] = min(1, max(image[cy][cx], add))

    for i in range(28):
        for j in range(28):
            pygame.draw.rect(screen, (int(image[i][j]*255),)*3, (j*size, i*size, size, size))

    prediction = model.predict(np.reshape(image,(784,1)))[0]
    if prediction != pre:
        print("\n"*50)
        print(prediction)
        pre = prediction

    pygame.display.flip()

        


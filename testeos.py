#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import math

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='patos')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de patos
filtro_1 = np.array([0, 251, 149]) 
filtro_2 = np.array([98, 255, 255]) 
MIN_AREA = 0

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


def red_line_detection(converted, frame):
    '''
    Detección de líneas rojas en el camino, esto es análogo a la detección de duckies,
    pero con otros filtros, notar también, que no es necesario aplicar houghlines en este caso
    '''
    # Se asume que no hay detección
    detection = False


    # Implementar filtros

    # Filtros de color para detectar líneas rojas
    low_red = np.array ([175, 120, 140])
    high_red = np.array([185, 240, 255])

    #Filtrar colores de la imagen en el rango utilizando 
    mask_red = cv2.inRange(converted, low_red, high_red)

    # Bitwise-AND mask and original
    segment_image = cv2.bitwise_and(converted, converted, mask= mask_red)

    #Vuelta al espacio BGR
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 

    # Kernel para segmentación
    kernel = np.ones((5,5),np.uint8)
    # Operacion morfologica erode
    image_out = cv2.erode(mask_red, kernel, iterations = 2)    
    # Operacion morfologica dilate
    image_out = cv2.dilate(image_out, kernel, iterations = 10)

    # Observar la imagen post-opening
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask= image_out)
    segment_image_post_opening =  cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    # y buscar los contornos
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar por area minima
        if w*h > 0:
            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,0,250), 3)

            # a la detección, además, Si hay detección, detection = True
            if w*h >= 500*100:
                detection = True

    cv2.imshow("Linea roja filtro", segment_image_post_opening)
    cv2.imshow("Linea roja detecciones", frame)

    return detection

# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

red_detection = False
def update(dt):

    """
    Funcion que se llama en step.
    """
    # Aquí se controla el duckiebot
    action = np.array([0.0, 0.0])
    global red_detection
    print (red_detection)
    if key_handler[key.UP] and red_detection == True:
        action[0]-=0.44
        red_detection = False
    if key_handler[key.UP] and red_detection == False:
        action[0]+=0.44
    if key_handler[key.DOWN]:
        action[0]-=0.44
    if key_handler[key.LEFT]:
        action[1]+=1
    if key_handler[key.RIGHT]:
        action[1]-=1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    # aquí se obtienen las observaciones y se setea la acción
    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()

    # Detección de patos
    # El objetivo es detectar los patos ajustando los valores del detector
    # obs = obs/255.0
    obs = obs.astype(np.uint8)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    #Cambiar tipo de color de RGB a HSV
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    red_detection = red_line_detection(converted, frame)

    def eliminate_half(image):
        height, width = image.shape[:2]
        # Create a new array of zeros with the same shape as the original image
        new_image = np.zeros_like(image)
        # Copy the relevant half of the original image into the new array
        new_image[height//2:height, :] = image[height//2:height, :]
        return new_image
    
    #Eliminar mitad superior de la imagen
    converted = eliminate_half(converted)


    obs_BGR = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    low_yellow = np.array ([0, 70, 150])
    high_yellow = np.array([30, 240, 240])
    low_white = np.array([0, 0, 140])
    high_white = np.array([170, 40, 255])

    #Funcion que filtra colores, aplica canny y detecta líneas de color en base a canales BGR
    def detector_lineas(low_array, high_array, color):
        #Filtrar colores de la imagen en el rango utilizando 
        mask = cv2.inRange(converted, low_array, high_array)

        # Bitwise-AND mask and original
        segment_image = cv2.bitwise_and(converted, converted, mask= mask)

        #Vuelta al espacio BGR
        image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)

        #Se añade detector de bordes canny
        bordes = cv2.Canny(image, 250, 300, None, 3)

        #Se añade detector de lineas Hough 
        lineas = cv2.HoughLines(bordes,1,np.pi/180,90)

       #Condicional para que no crashee al no encontrar lineas
        if type(lineas) == np.ndarray:
            # Se crean arrays que almacenen theta y rho de la línea de interés
            # array[0] = theta, array[1] = rho
            white_right = [[],[]]
            white_left = [[],[]]
            yellow = [[],[]]
            for linea in lineas:
                rho,theta = linea[0]
                # Si la línea es blanca se analiza si está a la izquierda o derecha del duckiebot
                if color == (0, 0, 255):
                    # Theta se mueve entre 0 y pi = 3.14, con pi/2 = 1.57
                    # Si la línea está a la derecha, se cumple lo siguiente
                    if (theta < 3.00 and theta > 1.70):
                        white_right[0].append(theta)
                        white_right[1].append(rho)
                    # Si está a la izquierda, se cumple lo siguiente
                    elif (theta > 0.20 and theta < 1.40):
                        white_left[0].append(theta)
                        white_left[1].append(rho)
                    # Si theta está entre 1.40 y 1.70, se asume que está horizontal en frente del duckiebot
                # Si la línea es amarilla, se almacenan sus valores no horizontales
                if color == (255, 0, 0):
                        if not (theta > 1.40 and theta < 1.70):
                            yellow[0].append(theta)
                            yellow[1].append(rho)
            
            new_lines = [white_right, white_left, yellow]
            for line in new_lines:
                # Se revisa si es que se encontró una línea de las buscadas
                if line != [[],[]]:
                    # Ahora se calcula el valor promedio de theta y rho para las líneas detectadas
                    theta = np.mean(line[0])
                    rho = np.mean(line[1])
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(obs_BGR,(x1,y1),(x2,y2),color,3)

        return bordes

    image_y = detector_lineas(low_yellow, high_yellow, (255, 0, 0))
    image_w = detector_lineas(low_white, high_white, (0, 0, 255))

    #Se juntan ambas imágenes filtradas
    image_yw = cv2.bitwise_or(image_y, image_w)

    bordes_resize = cv2.resize(image_yw, (480, 360))

    obs_resize = cv2.resize(obs_BGR, (480, 360))

    #cv2.imshow("Canny", bordes_resize)
    #cv2.imshow("Imagen segmentada", obs_resize)


    cv2.waitKey(1)
    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
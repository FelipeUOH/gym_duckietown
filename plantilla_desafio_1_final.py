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
window_filter_name = "filtro"

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

max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_filter_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_filter_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_filter_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_filter_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_filter_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_filter_name, high_V)


def red_alert(frame):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,2] = 90
    blend = cv2.addWeighted(frame, 0.5, red_img, 0.5, 0)

    return blend

cv2.namedWindow(window_filter_name)
cv2.createTrackbar(low_H_name, window_filter_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_filter_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_filter_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_filter_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_filter_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_filter_name , high_V, max_value, on_high_V_thresh_trackbar)


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


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


boleano=False
def update(dt):
    """
    Funcion que se llama en step.
    """
    # Aquí se controla el duckiebot
    action = np.array([0.0, 0.0])
    global boleano
    if key_handler[key.UP] and boleano==False:
        action[0]+=0.44
    if key_handler[key.UP] and boleano==True:
        action[0]+=0
        boleano=False
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
    # El objetivo de hoy es detectar los patos ajustando los valores del detector
    # obs = obs/255.0
    obs = obs.astype(np.uint8)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    # Filtro por color https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    
    #Cambiar tipo de color de RGB a HSV
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

    # Filtrar colores de la imagen en el rango utilizando 
    mask = cv2.inRange(converted, np.array([low_H, low_S, low_V]), np.array([high_H, high_S, high_V]))

    # Una vez que se hayan encontrado los parametros de la mascara, se debe reemplazar en las variables de la linea 25 y
    # ejecutar la siguiente linea comentando la anterior

    mask_duckie = cv2.inRange(converted, filtro_1, filtro_2)

    # Bitwise-AND mask and original 
    segment_image = cv2.bitwise_and(converted, converted, mask= mask_duckie)
    
    
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)

    # Esto corresponde a hacer un Opening
    # https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
    #Operacion morfologica erode
    image_out = cv2.erode(mask_duckie, kernel, iterations = 2)    
    #Operacion morfologica dilate
    image_out = cv2.dilate(image_out, kernel, iterations = 10)

    # https://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html
    contours, hierarchy = cv2.findContours(image_out, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Observar la imagen post-opening
    segment_image_post_opening = cv2.bitwise_and(converted, converted, mask= image_out)
    segment_image_post_opening =  cv2.cvtColor(segment_image_post_opening, cv2.COLOR_HSV2BGR)

    for cnt in contours:
        # Obtener rectangulo
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtrar por area minima
        if w*h > MIN_AREA:

            x2 = x + w  # obtener el otro extremo
            y2 = y + h
            # Dibujar un rectangulo en la imagen
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,0,250), 3)
        if w*h>= 140*170:
            print('red alert')
            boleano=True
            frame=red_alert(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (00, 185)
            color = (0, 0, 255)
            thickness = 2
            frame = cv2.putText(frame, 'ALERTA CUIDADO CON PATRICIO', org, font, 1,
                  color, thickness, cv2.LINE_AA, False) 


            

    cv2.imshow(window_filter_name, image)
    cv2.imshow("Imagen segmentada", segment_image_post_opening)
    cv2.imshow("Detecciones", frame)
    cv2.waitKey(1)
    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
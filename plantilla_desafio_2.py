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


def red_alert(frame):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,2] = 90
    blend = cv2.addWeighted(frame, 0.5, red_img, 0.5, 0)

    return blend


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

MIN_AREA = 0

def update(dt):
    """
    Funcion que se llama en step.
    """
    # Aquí se controla el duckiebot
    action = np.array([0.0, 0.0])
    global boleano
    if key_handler[key.UP]:
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

    # Detección lineas
    obs = obs.astype(np.uint8)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()
    
    #Funcion extraída de ayudantía 3 que corta una imagen a la mitad
    #Puede ser de utilidad para que el detector de líneas no fastidie detectando cosas fuera del camino
    #Función de ayudantía borra mitad de la imagen a eleccion, pero para este caso se buscará eliminar la mitad superior
    def eliminate_half(image):
        height, width = image.shape[:2]
        # Create a new array of zeros with the same shape as the original image
        new_image = np.zeros_like(image)
        # Copy the relevant half of the original image into the new array
        new_image[height//2:height, :] = image[height//2:height, :]
        return new_image
    
    #Eliminar mitad superior de la imagen
    converted = eliminate_half(obs)

    #Cambiar tipo de color de RGB a HSV
    converted = cv2.cvtColor(converted, cv2.COLOR_RGB2HSV)
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
        bordes = cv2.Canny(image, 250, 300)

        #Se añade detector de lineas Hough 
        lineas = cv2.HoughLines(bordes,1,np.pi/180,100)

       #Condicional para que no crashee al no encontrar lineas
        if type(lineas) == np.ndarray:
            for linea in lineas:
                rho,theta = linea[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(obs_BGR,(x1,y1),(x2,y2),color,2)

        return bordes

    image_y = detector_lineas(low_yellow, high_yellow, (255, 0, 0))
    image_w = detector_lineas(low_white, high_white, (0, 0, 255))

    #Se juntan ambas imágenes filtradas
    image_yw = cv2.bitwise_or(image_y, image_w)

    bordes_resize = cv2.resize(image_yw, (480, 360))

    obs_resize = cv2.resize(obs_BGR, (480, 360))

    cv2.imshow("Canny", bordes_resize)
    cv2.imshow("Imagen segmentada", obs_resize)

    cv2.waitKey(1)
    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
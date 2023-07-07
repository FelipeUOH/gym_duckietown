#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import math
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import time # Situaciones extremas requieren medidas desesperadas

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de lineas blancas
white_filter_1 = np.array([0, 0, 140])
white_filter_2 = np.array([170, 40, 255])

# Filtros para el detector de lineas amarillas
yellow_filter_1 = np.array([0, 70, 150])
yellow_filter_2 = np.array([30, 240, 240])
window_filter_name = "filtro"

# Constantes
DUCKIE_MIN_AREA = 140*170 #editar esto si es necesario
RED_LINE_MIN_AREA = 500*100 #editar esto si es necesario
RED_COLOR = (0,0,255)
MAX_DELAY = 20
MAX_TOLERANCE = 50

# Variables globales
last_vel = 0.44
delay = -1
tolerance = -1

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render(mode="top_down")


# Funciones interesantes para hacer operaciones interesantes
def box_area(box):
    return abs(box[2][0] - box[0][0]) * abs(box[2][1] - box[0][1])

def bounding_box_height(box):
    return abs(box[2][0] - box[0][0])

def get_angle_degrees2(x1, y1, x2, y2):
    return get_angle_degrees(x1, y1, x2, y2) if y1 < y2 else get_angle_degrees(x2, y2, x1, y1)
    
def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def get_angle_radians(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val


def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)

    return x, y

def yellow_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea amarilla :
    si su ángulo es cercano a 0 o 180, así como si es cercano a recto.detectada
    '''
    angle = get_angle_degrees2(x1, y1, x2, y2)
    return (angle < 30 or angle > 160) or (angle < 110 and angle > 90)

def white_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea blanca detectada:
    si se encuentra en el primer, segundo o tercer cuadrante, en otras palabras,
    se retorna False solo si la línea está en el cuarto cuadrante. (Y únicamente en el cuarto cuadrante ""ILÓGICO"")
    '''
    return (min(x1,x2) < 320) or (min(y1,y2) < 320)


def eliminate_half(image):
    '''
    Funcion que elimina la mitad superior de la imagen para no detectar cosas innecesarias
    '''
    height, width = image.shape[:2]
    # Create a new array of zeros with the same shape as the original image
    new_image = np.zeros_like(image)
    # Copy the relevant half of the original image into the new array
    new_image[height//2:height, :] = image[height//2:height, :]
    return new_image


def duckie_detection(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''

    # Se asume que no hay detección
    detection = False
    angle = 0

    '''
    Para lograr la detección, se puede utilizar lo realizado en el desafío 1
    con el freno de emergencia, aunque con la diferencia que ya no será un freno,
    sino que será un método creado por ustedes para lograr esquivar al duckie.
    '''

    # Implementar filtros

    # Filtrar colores de la imagen en el rango utilizando 
    mask_duckie = cv2.inRange(converted, np.array([0, 251, 149]), np.array([98, 255, 255]))

    # Bitwise-AND mask and original 
    segment_image = cv2.bitwise_and(converted, converted, mask= mask_duckie)

    # Realizar la segmentación con operaciones morfológicas (erode y dilate)

    # Kernel para segmentación
    kernel = np.ones((5,5),np.uint8)
    # Operacion morfologica erode
    image_out = cv2.erode(mask_duckie, kernel, iterations = 2)    
    # Operacion morfologica dilate
    image_out = cv2.dilate(image_out, kernel, iterations = 10)

    # Observar la imagen post-opening en BGR
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
            cv2.rectangle(frame, (int(x), int(y)), (int(x2),int(y2)), (0,0,255), 3)

            # a la detección, además, dentro de este for, se establece la detección = verdadera
            detection = True
            if w*h >= DUCKIE_MIN_AREA:
                # además del ángulo de giro angle = 'ángulo'
                angle = 15

    # Mostrar ventanas con los resultados
    cv2.imshow("Patos filtro", segment_image_post_opening)
    cv2.imshow("Patos detecciones", frame)

    return detection, angle


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
            if w*h >= RED_LINE_MIN_AREA:
                detection = True

    # Mostrar ventanas con los resultados

    cv2.imshow("Linea roja filtro", segment_image_post_opening)
    cv2.imshow("Linea roja detecciones", frame)

    return detection


def get_line(converted, filter_1, filter_2, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo
    del filtro aplicado, y de qué color trata, si es "white"
    y se cumplen las condiciones entonces gira a la izquierda,
    si es "yellow" y se cumplen las condiciones girar a la derecha.
    '''
    converted = eliminate_half(converted)
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)

    # Se define el return 'lado, angle' para evitar error 'local variable referenced before assignment'
    lado = None
    angle = None
    
    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)
    image_lines = cv2.erode(image, kernel, iterations = 2)    

    # Detectar líneas
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 150, 200, None, 3)
   
    # Detectar lineas usando houghlines y lo aprendido en el desafío 2.

    # Se añade detector de lineas Hough 
    lineas = cv2.HoughLines(edges,1,np.pi/180,85)

    # Condicional para que no crashee al no encontrar lineas
    if type(lineas) == np.ndarray:
        # Se crean arrays que almacenen theta y rho de la línea de interés
        # array[0] = theta, array[1] = rho
        white_right = [[],[]]
        white_left = [[],[]]
        yellow = [[],[]]
        for linea in lineas:
            rho,theta = linea[0]
            # Si la línea es blanca se analiza si está a la izquierda o derecha del duckiebot
            if line_color == 'white':
                # Theta se mueve entre 0 y pi = 3.14, con pi/2 = 1.57
                # Si la línea está a la derecha, se cumple lo siguiente
                if (theta < 3.10 and theta > 1.60): # Valores se pueden ajustarse
                    white_right[0].append(theta)
                    white_right[1].append(rho)
                # Si está a la izquierda, se cumple lo siguiente
                elif (theta > 0.10 and theta < 1.50): # Valores pueden ser ajustados
                    white_left[0].append(theta)
                    white_left[1].append(rho)
                # Si theta está entre 1.40 y 1.70, se asume que está horizontal en frente del duckiebot
            # Si la línea es amarilla, se almacenan sus valores no horizontales
            if line_color == 'yellow':
                    if not (theta > 1.40 and theta < 1.70):
                        yellow[0].append(theta)
                        yellow[1].append(rho)

        # Para evitar confusiones, la linea blanca derecha será la prioritaria
        # Si hay linea blanca derecha, se obvia linea blanca izquierda
        if white_right != [[],[]]:
            new_lines = [white_right, yellow]
        # Si no hay linea blanca derecha, se considera linea blanca izquiera (si es que hay)
        elif white_right == [[],[]]:
            new_lines = [white_left, yellow]
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

                cv2.line(image_lines,(x1,y1),(x2,y2),(0,0,255),3)

                # Se cubre cada color por separado, tanto el amarillo como el blanco
                # Con esto, ya se puede determinar mediante condiciones el movimiento del giro del robot.
                # Por ejemplo, si tiene una linea blanca muy cercana a la derecha, debe doblar hacia la izquierda
                # y viceversa.
                angle = get_angle_degrees2(x1,y1,x2,y2) #Valor entre 0 y 180
                # 0 y 180 representan una línea horizontal
                # 90 representa una línea vertical
                # Lineas a la derecha varian entre 0 y 90
                # Lineas a la izquierda varian entre 90 y 180

                if (angle >= 0 and angle < 90): # Valores se pueden ajustar
                    lado = 'der'
                elif (angle >= 90 and angle < 180):
                    lado = 'izq'

    cv2.imshow(line_color, image_lines)
    return lado, angle


def line_follower(vel, angle, obs):
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    # Se definen variables por defecto nuevamente para evitar 'local variable referenced before assignment'
    vel = 0.44
    new_angle_white = 0
    new_angle_yellow = 0
    # Variable que determina detecciones previas de linea roja
    pre_detection = False

    # Detección de duckies
    detection_duckie, _ = duckie_detection(obs=obs, converted=converted, frame=frame)
    '''
    Implementar evasión de duckies en el camino, variado la velocidad angular del robot
    '''
    # Evasion de duckies con velocidad
    # La idea es que si se detecta un duckie, el duckiebot debe pasar a la pista izquierda
    # Para esto se modifican los parámetros de navegación autónoma más adelante en 'new_angle'


    # Detección de líneas rojas
    detection_red = red_line_detection(converted=converted, frame=frame)
    ''' 
    Implementar detención por un tiempo determinado del duckiebot
    al detectar una linea roja en el camino, luego de este tiempo,
    el duckiebot debe seguir avanzando
    '''
    
    # Detección retorna un booleano que es True si linea roja está en zona crítica
    # Si hay detección y no habían detecciones previas, el vehículo se detiene
    if detection_red:
        print(detection_red, pre_detection)
    if detection_red == True and pre_detection == False:
        # Se determina detección previa para no volver a considerar detecciones
        pre_detection = True
        detection_red = False
        # El duckiebot se detiene por tres segundos
        time.sleep(3)

    # Si hay detección y ya hubo una detección previa, no se considera
    elif detection_red == True and pre_detection == True:
        detection_red = False

    # Si hay una detección previa pero no detecciones actuales, se reinicia la variable
    elif detection_red == False and pre_detection == True:
        pre_detection = False # Con esto debería seguir al pasar la linea


    # Obtener el ángulo propuesto por cada color
    lado_white, angle_white = get_line(converted, white_filter_1, white_filter_2, "white")
    lado_yellow, angle_yellow = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")


    '''
    Implementar un controlador para poder navegar dentro del mapa con los ángulos obtenidos
    en las líneas anteriores
    '''

    # Las reglas de conducción en el caso de tener pista derecha despejada
    if detection_duckie == False:
    # Pruebas mostraron que linea a la derecha debe ser 45 e izquierda 135 para avance óptimo
    # Por ejemplo, si duckiebot está inclinado hacia la derecha, el ángulo de línea derecha será menor
    # Deberá aumentar moviendose a la izquierda
        # Se detecta que el ángulo obtenido no sea NoneType (que no encontró línea) 
        if angle_white != None:
            if lado_white == 'der':
                # Condicional para asegurar que el ángulo está en el rango deseado
                if 0 < angle_white < 45:
                    new_angle_white = 1
                elif 45 < angle_white < 90:
                    new_angle_white = -1
            # Si linea blanca está en el lado izquierdo, el bot debe regresar al carril derecho
            elif lado_white == 'izq':
                new_angle_white = -3
        # Se aplica la misma condición para ángulo de línea amarilla
        if angle_yellow != None:
            # Si linea amarilla está en el lado izquierdo, se busca un ángulo de 135 grados
            if lado_yellow == 'izq':
                # Condicional para asegurar que el ángulo está en el rango deseado
                # Si el duckie va en línea recta, se da un angulo menor a 150 grados (estimados)
                if 90 < angle_yellow < 135:
                    new_angle_yellow = 1
                # Si el duckie está en curva cerrada, debe girar lento para no salirse de la ruta
                elif 135 < angle_yellow < 180:
                    new_angle_yellow = -1
            # Si linea amarilla está en el lado derecho, el bot debe regresar al carril derecho
            elif lado_yellow == 'der':
                new_angle_yellow = -2
        # Se prioriza el movimiento respecto a blancas por sobre amarillas

    # Si es que se detecta un duckie en el camino, debe cambiarse de pista
    elif detection_duckie == True:
        if angle_white != None:
        # Al cambiar de pista, se invierten las reglas de manejo para lineas blanca y amarilla
        # Suponiendo que duckie esté en el carril derecho, duckiebot debe moverse al carril izquierdo 
            if lado_white == 'der':
                new_angle_white = 2
            # Si linea blanca está en el lado izquierdo, su ángulo debe ser de 135 grados
            elif lado_white == 'izq':
                # Condicional para asegurar que el ángulo está en el rango deseado
                if 90 < angle_white < 135:
                    new_angle_white = 1
                elif 135 < angle_white < 180:
                    new_angle_white = -1
        if angle_yellow != None:
            # Si linea amarilla está en el lado izquierdo, se debe pasar al carril izquierdo
            if lado_yellow == 'izq':
                new_angle_yellow = 2
            # Si linea amarilla está en el lado derecho, se busca un ángulo de 45 grados
            elif lado_yellow == 'der':
                # Condicional para asegurar que el ángulo está en el rango deseado
                # Si el duckie está en curva cerrada, se da un angulo menor a 30 grados (estimados)
                # Debe girar lentamente para no salirse de la ruta
                if 0 < angle_yellow < 30:
                    new_angle_yellow = 0.5
                # Si el duckie va en línea recta, se da un angulo mayor a 30 grados
                elif 30 < angle_yellow < 45: 
                    new_angle_yellow = 1
                elif 45 < angle_yellow < 90:
                    new_angle_yellow = -1

    new_angle = new_angle_white + new_angle_yellow

    return np.array([vel, new_angle]) # Implementar nuevo ángulo de giro controlado

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    global action
    # Aquí se controla el duckiebot
    if key_handler[key.UP]:
        action[0] += 0.44
    if key_handler[key.DOWN]:
        action[0] -= 0.44
    if key_handler[key.LEFT]:
        action[1] += 1
    if key_handler[key.RIGHT]:
        action[1] -= 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    ''' Aquí se obtienen las observaciones y se setea la acción
    Para esto, se debe utilizar la función creada anteriormente llamada line_follower,
    la cual recibe como argumentos la velocidad lineal, la velocidad angular y 
    la ventana de la visualización, en este caso obs.
    Luego, se setea la acción del movimiento implementado con el controlador
    con action[i], donde i es 0 y 1, (lineal y angular)
    '''


    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    if done:
        print('done!')
        env.reset()
        env.render(mode="top_down")

    cv2.waitKey(1)
    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
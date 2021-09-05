import base64
import numpy as np
import socketio
import eventlet.wsgi
import eventlet
from PIL import Image
from flask import Flask
from io import BytesIO
import tensorflow as tf
from tensorflow.keras.models import load_model
from time import time
import cv2

tf.config.set_visible_devices([], 'GPU')

sio = socketio.Server()
app = Flask(__name__)
MAX_SPEED = 16
MIN_SPEED = 15
speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        print(data)

        image = np.asarray(image)
        image = image[60:160,0:360]
        # ShowImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # cv2.imshow('win' ,ShowImg)
        #
        # if cv2.waitKey(1)==ord(' '):
        #     pass

        image = np.array([image])

        t0 = time()
        steering_angle = float(model.predict(image, batch_size=1))
        t1 = time()

        print(f"Time Taken : {t1-t0}")

        global speed_limit
        if speed > speed_limit:
             speed_limit = MIN_SPEED  # slow down
        else:
             speed_limit = MAX_SPEED
        throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

        print('{} {} {}'.format(steering_angle, throttle, speed))
        send_control(steering_angle, throttle)
    else:
        print("Data not Found...")


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    model = load_model('CarModel2.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


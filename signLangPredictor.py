from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
from cv2 import circle
import numpy as np
import mediapipe as mp
import time
from matplotlib import pyplot as plt
import os

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Keypoints Using MP Holistic
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 255, 121), thickness=1, circle_radius=1))
    return image


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, face, lh, rh])


# Setup Folders For Collection
DATA_PATH = os.path.join('MP_DATA')

actions = np.array(['hello', 'thanks', 'iloveyou'])

no_sequences = 30

sequence_length = 30

# for action in actions:
#    for sequence in range(no_sequences):
#        try:
#            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#        except:
#            pass


#cap = cv2.VideoCapture(0)

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.7) as holistic:

#    for action in actions:

#        for sequence in range(sequence_length):

#            for frame_num in range(sequence_length):

# Read Feed
#                success, image = cap.read()

# Make Detections

#                image, results = mediapipe_detection(image, holistic)
# print(results)
#                draw_styled_landmarks(image, results=results)

#                if frame_num == 0:
#                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
#                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(
#                        action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
# Show to screen
#                    cv2.imshow("Live Feed", image)
#                    cv2.waitKey(1)

#                else:
#                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(
#                        action, sequence), (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
#                    cv2.imshow("Live Feed", image)

#cv2.imshow("Live Feed", img)
#                keypoints = extract_keypoints(results)
#                npy_path = os.path.join(
#                    DATA_PATH, action, str(sequence), str(frame_num))
#                np.save(npy_path, keypoints)

#                if cv2.waitKey(10) & 0xFF == ord('q'):
#                    break
#    cap.release()
#    cv2.destroyAllWindows()

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(
                sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

res = [.7, .2, .1]
print(actions[np.argmax(res)])

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
print(model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback]))


# ......................................SAVE WEIGHTS.................................
# model.save('action.h5')
#del model
# model.load_weights('action.h5')

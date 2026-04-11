from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras import backend as K

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Fix for TF1 session/graph issue in Flask ─────────────────────────────
graph = tf.get_default_graph()

def build_cnn_model():
    model = Sequential()
    model.add(Convolution2D(32, (1, 1), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Convolution2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model

def build_siamese_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, (1, 1), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Convolution2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model

# Load models at startup
cnn_model = build_cnn_model()
cnn_model.load_weights('model/cnn_weights.hdf5')
print("CNN model loaded.")

feature_extractor = Model(cnn_model.inputs, cnn_model.layers[-2].output)

# Warm up to get output shape
sample = np.zeros((1, 32, 32, 3))
with graph.as_default():
    sample_features = feature_extractor.predict(sample)

siamese_input_shape = (16, 16, 1)
siamese_model = build_siamese_model(siamese_input_shape)
siamese_model.load_weights('model/siamese_weights.hdf5')
print("Siamese model loaded.")

# Save session after loading
session = K.get_session()

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def extract_features(img):
    with graph.as_default():
        K.set_session(session)
        features = feature_extractor.predict(img)
    features = features.reshape(1, 16, 16, 1)
    return features

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Signature Recognition API is running'})

@app.route('/verify', methods=['POST'])
def verify():
    if 'reference' not in request.files or 'test' not in request.files:
        return jsonify({'error': 'Please upload both reference and test signature images.'}), 400

    ref_bytes = request.files['reference'].read()
    test_bytes = request.files['test'].read()

    ref_img = preprocess_image(ref_bytes)
    test_img = preprocess_image(test_bytes)

    ref_features = extract_features(ref_img)
    test_features = extract_features(test_img)

    with graph.as_default():
        K.set_session(session)
        ref_pred = siamese_model.predict(ref_features)
        test_pred = siamese_model.predict(test_features)

    ref_flat = ref_features.flatten()
    test_flat = test_features.flatten()
    cosine_sim = float(np.dot(ref_flat, test_flat) / (np.linalg.norm(ref_flat) * np.linalg.norm(test_flat) + 1e-10))

    THRESHOLD = 0.95
    if cosine_sim >= THRESHOLD:
        result = 'Real'
        confidence = round(cosine_sim * 100, 2)
    else:
        result = 'Forged'
        confidence = round((1 - cosine_sim) * 100, 2)

    return jsonify({
        'result': result,
        'confidence': confidence,
        'similarity': round(cosine_sim, 4)
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
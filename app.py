import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from flask import Flask, jsonify, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow_addons.metrics import F1Score
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Đăng ký custom metric
tf.keras.utils.get_custom_objects().update({'F1Score': F1Score})

# Đường dẫn đến thư mục chứa file weights.best.hdf5
weights_directory = 'weights'
weights_filename = 'weights.best.hdf5'
weights_path = os.path.join(weights_directory, weights_filename)

# Kiểm tra và tạo thư mục temp nếu chưa tồn tại
temp_directory = 'temp'
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

# Khai báo biến global để lưu trữ mô hình
model = None

def load_model_function():
    global model
    if model is None:
        model = load_model(weights_path)
    return model

@app.route('/')
def home():
    return "Welcome to the homepage!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận ảnh từ yêu cầu POST
        file = request.files['image']

        # Lưu trữ file tạm thời
        temp_filename = secure_filename(file.filename)
        temp_filepath = os.path.join(temp_directory, temp_filename)
        file.save(temp_filepath)

        # Load mô hình từ hàm đã tạo
        load_model_function()

        # Tiền xử lý ảnh
        img = image.load_img(temp_filepath, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]

        # Dự đoán
        predictions = model.predict(img_array)

        # Lấy kết quả dự đoán
        if predictions[0][0] > 0.5:
            result = 'Recaptured'
        else:
            result = 'Original'

        # Xóa file tạm thời
        os.remove(temp_filepath)

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

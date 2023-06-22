import os
import sys
import time
from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging, log_path, LOG_FILE

from src.pipelines.train_pipeline import TrainingPipeline, DataTransformation
from src.pipelines.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train_model():
    try:
        train = TrainingPipeline()
        train.run_pipeline()
        return render_template('success.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods = ['POST','GET'])
def upload():
    try:
        if request.method == 'POST':
            predict_pipeline = PredictionPipeline() 
            predict_file_detail = predict_pipeline.run_pipeline()
            logging.info('Prediction completed successfully')
            return send_file(predict_file_detail.prediction_file_path,
                            download_name=predict_file_detail.prediction_file_path,
                            as_attachment=True)
        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e,sys)
    
@app.route('/logged')
def training_logger():
    try:
        
        LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)
        log_contents = []

        with open(LOG_FILE_PATH, 'r') as file:
            logs = file.readlines()
            filtered_logs = [log for log in logs if 'root' in log]
            log_contents.extend(filtered_logs)

        return render_template('logs.html', logs=log_contents)
    except Exception as e:
        raise CustomException(e,sys)


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
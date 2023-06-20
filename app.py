import os
import sys
from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging

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


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
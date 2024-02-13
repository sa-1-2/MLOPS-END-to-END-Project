from flask import Flask, render_template, jsonify, request
from mlops.pipeline.prediction_pipeline import PredictPipeline, CustomData
from mlops.logger.log import logging
from mlops.exception.exception import customexception
import os
import sys

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("index.html")


@app.route('/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            logging.info("gathering custom data values")
            carat = float(request.form.get("carat"))
            depth = float(request.form.get("depth"))
            table = float(request.form.get("table"))
            x = float(request.form.get("x"))
            y = float(request.form.get("y"))
            z = float(request.form.get("z"))
            cut = str(request.form.get("cut"))
            color = str(request.form.get("color"))
            clarity = str(request.form.get("clarity"))
            print(color, clarity, cut, carat, x, y, z, table, depth)
            data = CustomData(carat=carat, depth=depth, table=table, x=x, y=y,z=z, cut=cut, color=color, clarity=clarity)
        
            df_final = data.get_data_as_dataframe()
            logging.info("dataframe is made")

            prediction = PredictPipeline()
            pred = prediction.predict(df_final)
            result = round(pred[0], 2)
            logging.info("prediction is done")
            return render_template("result.html", final_result = result)
        
        except Exception as e:
            logging.info("Error occured while data gathering in app.py")
            raise customexception(e, sys)
        


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def fetch_answer(inputs):
    model = tf.keras.models.load_model('C:\\Users\\ACER\\Desktop\\use\\new_model_brent.keras')

    to_pred = np.array([[pd.to_datetime(inputs).map(pd.Timestamp.toordinal)]])
    preds = model.predict(to_pred)

    result = {
        'date': inputs,
        'price': float(preds.ravel()[0])
    }

    # Use standard JSON instead of Django JsonResponse
    return json.dumps(result)
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os



# Create a Flask app
app = Flask(__name__)
script_dir=os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, '/home/WaelAhmed/mysite/gbr_pipeline.pkl')
GBR_pipeline = joblib.load(model_path)

@app.route('/')
def home():
    return "Now Run Successfully......"


# Define an API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)

        new_data_transformed = GBR_pipeline.named_steps['preprocessor'].transform(data)

        prediction = GBR_pipeline.named_steps['regressor'].predict(new_data_transformed)

        return jsonify({'Prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})



# if __name__ == '__main__':
#     app.run()
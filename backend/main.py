from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os 
import joblib

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name = 'static')
rf_model = joblib.load('models/rf_model.joblib')
rf_scaler = joblib.load('models/rf_scaler.joblib')
result_encoder = joblib.load('models/result_encoder.joblib')
team_mapping = joblib.load('models/team_mapping.joblib')

@app.get('/', response_class = FileResponse)
async def serve_index():
    #Create a root directory
    return FileResponse('index.html')

@app.get('/predict')
async def predict_match(teamA: str, teamB: str):
    teamA_encoded = team_mapping.get(teamA)
    teamB_encoded = team_mapping.get(teamB)

    if teamA_encoded is None or teamB_encoded is None:
        return {'error': 'Unknown team'}
    
    input_data = [[teamA_encoded, teamB_encoded]]
    input_scaled = rf_scaler.transform(input_data)
    result_code = rf_model.predict(input_scaled)[0]
    result_label = result_encoder.inverse_transform([result_code])[0]
    prediction = rf_model.predict_proba(input_scaled)[0]
    print(result_encoder.classes_, prediction)
    return {
        teamA : 'home',
        teamB : 'away',
        'prediction': result_label,
        'probs': dict(zip(result_encoder.classes_, prediction))
        }

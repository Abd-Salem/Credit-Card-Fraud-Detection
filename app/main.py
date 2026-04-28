from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from load_configs import Config
from app.schemas.input_schemas import Transaction
from app.service.predictor import  ModelService
from app.service.predictor import prepare_input


# load model pipeline with best threshold
config = Config()
model = ModelService(config.MODELS['best_model']['model'], threshold=0.65)

app = FastAPI(title='Fraud-Detection-API')
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post('/predict')
def predict(transaction: Transaction):
    try:
        X = prepare_input(transaction.features)
        prob, pred = model.predict(X)

        return {
            'Fraud' : int(pred),
            'Probability' : float(prob),
            'Threshold' : model.threshold,
            'model_name': model.name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import time
import ai as ai
from pathlib import Path
import scraper


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://127.0.0.1:3000",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    map: str
    brawlers: List[str]
    first_pick: bool


class PickrateRequest(BaseModel):
    map: str


@app.get("/maps")
async def get_maps():
    return {"maps": ai.get_all_maps()}


@app.get("/brawlers")
async def get_brawlers():
    return {"brawlers": ai.get_all_brawlers()}


last_prediction_time = 0
PREDICTION_COOLDOWN = 0.1  # 0.1 second cooldown between predictions


@app.post("/pickrate")
async def retrieve_map_pickrates(request: PickrateRequest):
    global last_prediction_time
    current_time = time.time()

    if current_time - last_prediction_time < PREDICTION_COOLDOWN:
        raise HTTPException(status_code=429,
                            detail="Too many requests. Please wait "
                                   "before retrieving pickrate again.")

    last_prediction_time = current_time

    try:
        print(f"Pickrate request: Map: {request.map}")
        probabilities = ai.get_map_pickrate(request.map)
        print(f"Pickrate results: {probabilities}")
        return {"pickrate": probabilities}
    except Exception as e:
        print(f"Error during pickrate retrieval: {e}")
        raise HTTPException(status_code=500, detail="Pickrate retrieval "
                                                    "failed")


@app.post("/predict")
async def predict_brawlers(request: PredictionRequest):
    global last_prediction_time
    current_time = time.time()

    if current_time - last_prediction_time < PREDICTION_COOLDOWN:
        raise HTTPException(status_code=429,
                            detail="Too many requests. Please "
                                   "wait before predicting again.")

    last_prediction_time = current_time

    try:
        print(f"Prediction request: Map: {request.map}, "
              f"Brawlers: {request.brawlers}, "
              f"First Pick: {request.first_pick}")
        if (request.brawlers == []):
            probabilities = ai.get_map_winrate(request.map)
        else:
            brawler_dict = ai.get_brawler_dict(request.brawlers,
                                               request.first_pick)
            probabilities = ai.predict(brawler_dict, request.map, request.first_pick)
        print(f"Prediction results: {probabilities}")
        return {"probabilities": probabilities}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


BRAWLER_MAPPING_FILE = Path("./out/brawlers/brawler_supercell_id_mapping.json")

def load_brawler_mapping():
    try:
        with open(BRAWLER_MAPPING_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@app.get("/brawler-mapping")
async def get_brawler_mapping():
    mapping = load_brawler_mapping()
    if not mapping:
        raise HTTPException(status_code=404, detail="Brawler mapping not found")
    return JSONResponse(content=mapping)


@app.post("/update-brawler-mapping")
async def update_brawler_mapping():
    scraper.brawler_to_supercell_id_mapping()
    return {"message": "Brawler mapping updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)

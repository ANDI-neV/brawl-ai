from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json
import time
from pathlib import Path
import scraper
from ai import get_map_score, get_all_maps, get_all_brawlers, get_brawler_dict, get_map_pickrate, PlayerNotFoundError, get_filtered_brawlers
from inference import predict, reload_model
import configparser

app = FastAPI()

config = configparser.ConfigParser()
config.read('config.ini')
pi_host = config['Pi']['pi_host']
main_host = config['Pi']['main_host']
public_ip = config['Pi']['public_ip']
domain = config['Pi']['domain']
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3003",
                   "http://127.0.0.1:3003",
                   f"http://{pi_host}:3003",
                   f"http://{main_host}:3000",
                   f"http://{public_ip}:3003",
                   f"http://{domain}",
                    f"https://{domain}",
                    f"http://www.{domain}",
                    f"https://www.{domain}",
                   ],
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
    return {"maps": get_all_maps()}


@app.get("/brawlers")
async def get_brawlers():
    return {"brawlers": get_all_brawlers()}


last_prediction_time = 0
PREDICTION_COOLDOWN = 0  # 0.1 second cooldown between predictions


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
        probabilities = get_map_pickrate(request.map)
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
            probabilities = get_map_score(request.map)
        else:
            brawler_dict = get_brawler_dict(request.brawlers,
                                               request.first_pick)
            probabilities = predict(brawler_dict, request.map, request.first_pick)
        if not probabilities:
            return {"probabilities": {}}
        else:
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


class FilteredBrawlerRequest(BaseModel):
    player_tag: str
    min_level: int


@app.post("/filtered-player-brawlers")
async def filter_player_brawlers(request: FilteredBrawlerRequest):
    try:
        print(f"Player tag: {request.player_tag}, "
              f"Minimum brawler level: {request.min_level}")

        filtered_brawlers = get_filtered_brawlers(request.player_tag, request.min_level)

        return {"brawlers": filtered_brawlers}
    except PlayerNotFoundError:
        raise HTTPException(status_code=404, detail="Player tag not found")
    except Exception as e:
        print(f"Error during player brawler retrieval: {e}")
        raise HTTPException(status_code=500, detail="Player brawler retrieval failed")


@app.post("/reload-model")
async def reload_model_endpoint():
    try:
        reload_model()
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        print(f"Error during model reload: {e}")
        raise HTTPException(status_code=500, detail="Model reload failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7001)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from db import Database
import new_transformer_approach as ai

app = FastAPI()
db = Database()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    map: str
    brawlers: List[str]

@app.get("/maps")
async def get_maps():
    return {"maps": db.getAllMaps()}

@app.get("/brawlers")
async def get_brawlers():
    return {"brawlers": ai.get_all_brawlers()}

@app.post("/predict")
async def predict_brawlers(request: PredictionRequest):
    print(request.map, request.brawlers, request.first_pick)
    brawler_dict = ai.get_brawler_dict(request.brawlers, request.first_pick)
    probabilities = ai.predict(brawler_dict, request.map)
    return {"probabilities": probabilities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)
import requests
import configparser
from db import Database
        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.token = config["Credentials"]["api"]
    def get_player_battlelog(self, player_tag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        #print("Requesting from " + f"{self.url}/v1/players/%23{playerTag}/battlelog")
        response = requests.get(f"{self.url}/v1/players/%23{player_tag}/battlelog", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            if response.json()["reason"] == "notFound":
                print("Player not found: " + player_tag)
                return "notFound"

            else:
                return None
        
    def get_player_stats(self, player_tag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(f"{self.url}/v1/players/%23{player_tag}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            if response.status_code != 429 or response.status_code != 404:
                print(response.status_code)
            return None

    def get_brawler_information(self):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(f"{self.url}/v1/brawlers", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            if response.status_code != 429 or response.status_code != 404:
                print(response.status_code)
            return None

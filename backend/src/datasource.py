import requests

        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        self.token = "REDACTED"
    def getPlayerBattlelog(self, playerTag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        #print("Requesting from " + f"{self.url}/v1/players/%23{playerTag}/battlelog")
        response = requests.get(f"{self.url}/v1/players/%23{playerTag}/battlelog", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            if response.status_code != 429 or response.status_code != 404:
                print(response.status_code)
            return None
        
    def get_player_stats(self, playerTag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(f"{self.url}/v1/players/%23{playerTag}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            if response.status_code != 429 or response.status_code != 404:
                print(response.status_code)
            return None

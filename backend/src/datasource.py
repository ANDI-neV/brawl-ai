import requests

from settings import get_api_token
        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        self.token = get_api_token()
    def get_player_battlelog(self, player_tag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(
            f"{self.url}/v1/players/%23{player_tag}/battlelog",
            headers=headers,
            timeout=20,
        )
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            response_data = response.json()
            if response_data.get("reason") == "notFound":
                print("Player not found: " + player_tag)
                return "notFound"
            return None
        
    def get_player_stats(self, player_tag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(
            f"{self.url}/v1/players/%23{player_tag}",
            headers=headers,
            timeout=20,
        )
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            if response.status_code not in {404, 429}:
                print(response.status_code)
            return None

    def get_brawler_information(self):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(
            f"{self.url}/v1/brawlers",
            headers=headers,
            timeout=20,
        )
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            if response.status_code not in {404, 429}:
                print(response.status_code)
            return None

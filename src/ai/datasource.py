import requests

class BrawlStarsAPI:
    def __init__(self):
        self.url = "https://api.brawlify.com/v1/events"

    def get_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
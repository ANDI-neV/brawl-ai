import requests

class BrawlAPI:
    def __init__(self):
        self.url = "https://api.brawlify.com/v1/events"

    def get_data(self):
        response = requests.get(self.url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        self.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjJiNjY5OTlmLWFkNTUtNDJlZC1iYTMzLTRiM2QzNjdkNmQxOCIsImlhdCI6MTcxOTY5NTQ1Nywic3ViIjoiZGV2ZWxvcGVyL2YxYzg1ODczLTE5YWUtZTkwZi02N2QyLTU0ZWFlYmM3YmFmMSIsInNjb3BlcyI6WyJicmF3bHN0YXJzIl0sImxpbWl0cyI6W3sidGllciI6ImRldmVsb3Blci9zaWx2ZXIiLCJ0eXBlIjoidGhyb3R0bGluZyJ9LHsiY2lkcnMiOlsiMi4yMDYuMjAyLjE3OCIsIjc3LjQ3LjYyLjIyNCJdLCJ0eXBlIjoiY2xpZW50In1dfQ.A-wYJ6Ok8pMCShviykxGZFEUOzhthLt7pVAh-mgmnQdg5iJYI1OAm-2EjYSJ9QYE1JBfP5a3N3PtNID97yXPtg"
    def getPlayerStats(self, playerTag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        print("Requesting from" + f"{self.url}/v1/players/%23{playerTag}/battlelog")
        response = requests.get(f"{self.url}/v1/players/%23{playerTag}/battlelog", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            print(response.status_code)
            return None
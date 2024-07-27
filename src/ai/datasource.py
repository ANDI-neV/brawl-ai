import requests

        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        self.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6ImFhMmQzYmEyLWM3OGMtNGIxNi04N2U2LTI0OGJmOGQ5MTY0MiIsImlhdCI6MTcyMjA5Mjc0Mywic3ViIjoiZGV2ZWxvcGVyL2YzYjQwZTQ3LTQ3NGMtZGI2Ni0wNjBjLTA2MWQwN2QyNWU2NCIsInNjb3BlcyI6WyJicmF3bHN0YXJzIl0sImxpbWl0cyI6W3sidGllciI6ImRldmVsb3Blci9zaWx2ZXIiLCJ0eXBlIjoidGhyb3R0bGluZyJ9LHsiY2lkcnMiOlsiNzcuNC4yOS4xODEiXSwidHlwZSI6ImNsaWVudCJ9XX0.Ove3B7vgLEdRH473EgoHsLYwJYLq758kDoaaO3pdC8J7ikHs73mKPj-MtjocV1knf1SwfXIA1rXKLwgk-W5Snw"
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
            print(response.status_code)
            return None
        
    def getPlayerStats(self, playerTag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        response = requests.get(f"{self.url}/v1/players/%23{playerTag}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            # parse the ClientError Model
            print(response.status_code)
            return None

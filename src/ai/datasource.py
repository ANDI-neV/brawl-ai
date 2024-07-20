import requests

        
class DevBrawlAPI:
    # This uses a token
    def __init__(self):
        self.url = "https://api.brawlstars.com"
        self.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjhkZDViMmIyLTA1ZDItNDJmNC04MDdkLTg3MzgyNWI0NmYyYSIsImlhdCI6MTcyMTMzMTU0OSwic3ViIjoiZGV2ZWxvcGVyL2YxYzg1ODczLTE5YWUtZTkwZi02N2QyLTU0ZWFlYmM3YmFmMSIsInNjb3BlcyI6WyJicmF3bHN0YXJzIl0sImxpbWl0cyI6W3sidGllciI6ImRldmVsb3Blci9zaWx2ZXIiLCJ0eXBlIjoidGhyb3R0bGluZyJ9LHsiY2lkcnMiOlsiMi4yMDMuMTk3LjIyNyJdLCJ0eXBlIjoiY2xpZW50In1dfQ.H8wmDltmi0vfaqrDphmvhsDC1y_XKSR23Gv-yN_oO8G5uXrORRG2-Rz8YW3b8dkUUrvHK5efr6nZwrauJxMkBg"
    def getPlayerBattlelog(self, playerTag):
        headers = {
            "Authorization": f"Bearer {self.token}"
        }
        print("Requesting from " + f"{self.url}/v1/players/%23{playerTag}/battlelog")
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

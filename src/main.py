from ai.datasource import DevBrawlAPI
import json

# Leon Tag: 2QYUPPUG8

if __name__ == "__main__":
    api = DevBrawlAPI()
    player = api.getPlayerStats("2QYUPPUG8")
    parsed = json.dumps(player, indent=4)  
    print(parsed)


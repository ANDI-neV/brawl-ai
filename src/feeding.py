from ai.datasource import DevBrawlAPI
from db import Database
import json
import time
import random
from datetime import datetime




api = DevBrawlAPI()
db = Database()
random.seed(datetime.now().timestamp())
firstplayer = "2QYUPPUG8"



if(True): # collecting mode
    if db.get_players_count() == 0:
        playerapi = api.getPlayerStats(firstplayer)
        player = (
            playerapi["tag"][1:],
            playerapi["name"]
        )
        db.insert_player(player)

    while db.get_battles_count() < 20000:
        player = db.get_unchecked_player()
        playerTag = player[0]
        db.set_player_checked(playerTag)
        battlelog = api.getPlayerBattlelog(playerTag)
        if battlelog:
            for battle in battlelog["items"]:
                try:
                    battleTime = battle["battleTime"]
                    map = battle["event"]["map"]
                    mode = battle["battle"]["mode"]
                    allowedModes = ["gemGrab", "brawlBall", "heist", "bounty", "hotZone", "knockout"]
                    if mode in allowedModes:
                        a1 = battle["battle"]["teams"][0][0]["brawler"]["name"]
                        a2 = battle["battle"]["teams"][0][1]["brawler"]["name"]
                        a3 = battle["battle"]["teams"][0][2]["brawler"]["name"]
                        b1 = battle["battle"]["teams"][1][0]["brawler"]["name"]
                        b2 = battle["battle"]["teams"][1][1]["brawler"]["name"]
                        b3 = battle["battle"]["teams"][1][2]["brawler"]["name"]
                        result = battle["battle"]["result"]
                        unixtime = time.mktime(time.strptime(battleTime, "%Y%m%dT%H%M%S.000Z"))
                        unixtime = int(unixtime)
                        #print(unixtime)
                        #print(map)
                        #print(result)
                        if result == "victory":
                            result = 1
                        else:
                            result = 0
                        #print(a1, a2, a3)
                        #print(b1, b2, b3)
                        id = random.randint(0, 10000000000000)
                        #print("----")
                        db.insert_battle((id, unixtime, map, mode, a1, a2, a3, b1, b2, b3, result))

                        playerstags = [
                            battle["battle"]["teams"][0][0]["tag"],
                            battle["battle"]["teams"][0][1]["tag"],
                            battle["battle"]["teams"][0][2]["tag"],
                            battle["battle"]["teams"][1][0]["tag"],
                            battle["battle"]["teams"][1][1]["tag"],
                            battle["battle"]["teams"][1][2]["tag"]
                        ]
                        playersnames = [
                            battle["battle"]["teams"][0][0]["name"],
                            battle["battle"]["teams"][0][1]["name"],
                            battle["battle"]["teams"][0][2]["name"],
                            battle["battle"]["teams"][1][0]["name"],
                            battle["battle"]["teams"][1][1]["name"],
                            battle["battle"]["teams"][1][2]["name"]
                        ]
                        for i in range(6):
                            enemyplayer = (
                                playerstags[i][1:],
                                playersnames[i]
                            )
                            db.insert_player(enemyplayer)
                except Exception as e:
                    print(e)
                    print(battle)

                

        time.sleep(0.3)
        

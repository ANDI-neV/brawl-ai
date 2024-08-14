from backend.datasource import DevBrawlAPI
from db import Database
import json
import time
import random
from datetime import datetime
import threading

api = DevBrawlAPI()
db = Database()
random.seed(datetime.now().timestamp())
firstplayer = "2QYUPPUG8"



def pushBattle(battle):
    db.insert_battle(battle)

def getNewEntries():
    player = db.get_unchecked_player()
    playerTag = ""
    try:
        playerTag = player[0]
    except Exception as e:
        print(player)
    db.set_player_checked(playerTag)
    battlelog = api.getPlayerBattlelog(playerTag)
    if battlelog:
        for battle in battlelog["items"]:
            try:
                if "type" not in battle["battle"]:
                    continue
                if battle["battle"]["type"] == "soloRanked":
                    battleTime = battle["battleTime"]
                    map = battle["event"]["map"]
                    mode = battle["battle"]["mode"]
                    a1 = battle["battle"]["teams"][0][0]["brawler"]["name"]
                    a2 = battle["battle"]["teams"][0][1]["brawler"]["name"]
                    a3 = battle["battle"]["teams"][0][2]["brawler"]["name"]
                    b1 = battle["battle"]["teams"][1][0]["brawler"]["name"]
                    b2 = battle["battle"]["teams"][1][1]["brawler"]["name"]
                    b3 = battle["battle"]["teams"][1][2]["brawler"]["name"]
                    result = battle["battle"]["result"]
                    unixtime = time.mktime(
                        time.strptime(battleTime, "%Y%m%dT%H%M%S.000Z"))
                    unixtime = int(unixtime)
                    if result == "victory":
                        result = 1
                    else:
                        result = 0
                    id = random.randint(0, 10000000000000)
                    db.insert_battle((id, unixtime, map, mode, a1, a2, a3, b1,
                                      b2, b3, result))
                    

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
                        enemyplayer = (playerstags[i][1:], playersnames[i])
                        db.insert_player(enemyplayer)
                    else:
                        continue
            except Exception as e:
                print(type(e))
                print(battle)
                
        db.commit()



if __name__ == "__main__":


    if True:
        print("Currently there are " + str(db.get_battles_count()) + " battles in the database")

        starttime = time.time()
        if db.get_players_count() == 0:
            playerapi = api.getPlayerStats(firstplayer)
            player = (playerapi["tag"][1:], playerapi["name"])
            db.insert_player(player)
        runtime = 1
        while (time.time() - starttime) < 60 * runtime:
            print("Time left: " + str((60 * runtime -
                                    (time.time() - starttime)) / 60) + " min")
            getNewEntries()

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

# 1. Get an unchecked player
# 2. Get the battlelog of that player
# 3. For each battle in the battlelog:
# 4. Get all the players in the battle
# 5. For each player in the battle:
# 6. Check if player can do ranked
# 7. If yes, add players to the database
# 8. If yes, add battle to the database
# 9. If no, skip the battle
# 10. Mark the player as checked
# 11. Repeat


def pushBattle(battle):
    db.insert_battle(battle)


def get_unchecked_player(count=1):
    return db.get_unchecked_player(count)


def get_players_from_battlelog(battle):
    players = []
    for team in battle["battle"]["teams"]:
        for player in team:
            players.append((player["tag"][1:], player["name"]))
    return players


def check_ranked_eligibility(playerstats):
    brawlers = playerstats["brawlers"]
    count = 0
    for brawler in brawlers:
        if brawler["power"] >= 9:
            count += 1
    if count >= 12:
        return True
    else:
        return False


class BattleLogsThread(threading.Thread):

    def __init__(self, batch):
        threading.Thread.__init__(self)
        self.batch = batch
        self.result = []

    def battleLogsThread(self):
        logs = []
        for playerTag in self.batch:
            #logs.append(api.getPlayerBattlelog(playerTag))
            for _ in range(3):
                logsvalue = api.getPlayerBattlelog(playerTag)
                if logsvalue is not None:
                    logs.append(logsvalue)
                    break
                time.sleep(0.2)
        self.result = logs

    def run(self):
        self.battleLogsThread()


def get_battlelogs(playerTags):
    logs = []
    '''for playerTag in playerTags:
        logs.append(api.getPlayerBattlelog(playerTag))'''
    batch_amount = 40
    batches = []
    playerscount = len(playerTags)
    for i in range(batch_amount):
        start = i * playerscount // batch_amount
        end = (i + 1) * playerscount // batch_amount
        batches.append(playerTags[start:end])
    threads = []
    for batch in batches:
        threads.append(BattleLogsThread(batch))
    for thread in threads:
        thread.start()
        time.sleep(1 / batch_amount)
    for thread in threads:
        thread.join()
    for thread in threads:
        logs += thread.result
    return logs


class PlayerStatsThread(threading.Thread):

    def __init__(self, batch):
        threading.Thread.__init__(self)
        self.batch = batch
        self.result = []

    def playerStatsThread(self):
        stats = []
        for player in self.batch:
            statsvalue = api.getPlayerStats(player)
            # check for NoneType
            for _ in range(3):
                if statsvalue is not None:
                    stats.append(statsvalue)
                    break
                time.sleep(0.2)
                #print("Retrying " + player)
                statsvalue = api.getPlayerStats(player)

        self.result = stats

    def run(self):
        self.playerStatsThread()


def get_player_stats(players):
    stats = []
    playerTags = players
    batch_amount = 40
    batches = []
    playerscount = len(playerTags)
    for i in range(batch_amount):
        start = i * playerscount // batch_amount
        end = (i + 1) * playerscount // batch_amount
        batches.append(playerTags[start:end])

    threads = []
    for batch in batches:
        threads.append(PlayerStatsThread(batch))
    for thread in threads:
        thread.start()
        time.sleep(1 / batch_amount)
    for thread in threads:
        thread.join()
    for thread in threads:
        stats += thread.result
    return stats


def checkBattlewithOnlyEligiblePlayers(battle, eligiblePlayers):
    try:
        for team in battle["battle"]["teams"]:
            for player in team:
                if player["tag"][1:] in eligiblePlayers:
                    continue
                else:
                    return False
        return True
    except Exception as e:
        print(e)
        print(json.dumps(battle))
        return False


def cycle():
    batchsize = 1000
    playerslist = []
    playerslist = get_unchecked_player(batchsize)
    playerTags = [player[0] for player in playerslist]
    starttime = time.time()
    battlelogs = get_battlelogs(playerTags)
    print("Battlelogs took: " + str(time.time() - starttime) + " seconds")
    battles = []
    for battlelog in battlelogs:
        for battle in battlelog["items"]:
            battles.append(battle)
    newPlayers = []
    filteredBattles = []
    for battle in battles:  # Get all players from the battles
        if "type" not in battle["battle"]:
            continue
        if battle["battle"]["type"] == None:
            continue
        if battle["battle"]["type"] != "soloRanked":
            continue
        filteredBattles.append(battle)
        newPlayers += get_players_from_battlelog(battle)
    battles = filteredBattles
    newPlayers = list(set(newPlayers))  # Remove duplicates
    playerTags = [player[0] for player in newPlayers]
    existingPlayers = db.get_if_players_exist(playerTags)
    playerTags = list(set(playerTags) - set(existingPlayers))
    starttime = time.time()
    playerStatsList = get_player_stats(playerTags)
    playerStatCount = len(playerStatsList)
    elgiblePlayers = []
    for playerStats in playerStatsList:
        if check_ranked_eligibility(playerStats):
            elgiblePlayers.append(playerStats["tag"][1:])

    print("Time taken: " + str(time.time() - starttime) + " seconds")
    print("Players: " + str(playerStatCount))
    print("Eligible Players: " + str(len(elgiblePlayers)))
    print("Battles: " + str(len(battles)))
    realBattles = []
    for battle in battles:
        if checkBattlewithOnlyEligiblePlayers(battle, elgiblePlayers):
            realBattles.append(battle)
    print("Real Battles: " + str(len(realBattles)))
    db.insert_many_players(newPlayers)
    for battle in realBattles:
        id = random.randint(0, 10000000000000)
        unixtime = time.mktime(
            time.strptime(battle["battleTime"], "%Y%m%dT%H%M%S.000Z"))
        unixtime = int(unixtime)
        sqlbattle = (id, unixtime, battle["event"]["map"],
                     battle["battle"]["mode"],
                     battle["battle"]["teams"][0][0]["brawler"]["name"],
                     battle["battle"]["teams"][0][1]["brawler"]["name"],
                     battle["battle"]["teams"][0][2]["brawler"]["name"],
                     battle["battle"]["teams"][1][0]["brawler"]["name"],
                     battle["battle"]["teams"][1][1]["brawler"]["name"],
                     battle["battle"]["teams"][1][2]["brawler"]["name"],
                     1 if battle["battle"]["result"] == "victory" else 0)
        pushBattle(sqlbattle)
    db.set_many_players_checked(playerTags)
    db.commit()



if __name__ == "__main__":

    if True:
        print("Currently there are " + str(db.get_battles_count()) +
              " battles in the database")
        print("Currently there are " + str(db.get_players_count()) +
                " players in the database")

        starttime = time.time()
        if db.get_players_count() == 0:
            playerapi = api.getPlayerStats(firstplayer)
            player = (playerapi["tag"][1:], playerapi["name"])
            db.insert_player(player)
        runtime = 15
        while (time.time() - starttime) < 60 * runtime:
            print("Time left: " + str((60 * runtime -
                                       (time.time() - starttime)) / 60) +
                  " min")
            cycle()
            print("Checked players: " +
                  str(db.get_checked_players_percentage()))
            print("Battles in database: " + str(db.get_battles_count()))
            print("Players in database: " + str(db.get_players_count()))
            

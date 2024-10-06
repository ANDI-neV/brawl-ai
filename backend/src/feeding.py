from datasource import DevBrawlAPI
from db import Database
import json
import time
import random
from datetime import datetime
import threading
from typing import Tuple, List, Dict, Any


class DevBrawlManager():
    """
    1. Get an unchecked player
    2. Get the battlelog of that player
    3. For each battle in the battlelog:
    4. Get all the players in the battle
    5. For each player in the battle:
    6. Check if player can do ranked
    7. If yes, add players to the database
    8. If yes, add battle to the database
    9. If no, skip the battle
    10. Mark the player as checked
    11. Repeat
    """
    def __init__(self):
        self.api = DevBrawlAPI()
        self.db = Database()
        random.seed(datetime.now().timestamp())
        self.first_player = "2QYUPPUG8"
        self.best_players = [
            "PR9U2JL", "VL0GPPJV", "2PLVJ0GLV", "8VJ8GJR2",
            "JGCCGY80", "PYPQUG80", "2ULUP8VG9", "22C0GV9CR",
            "R0GP8PQ98", "QLCJGQUP", "2PGGR8Y9P", "JQVQYVY",
            "9GJPJUQGJ", "8V92UYCJ", "9VJP20UYU", "LRLQPU09",
            "P2G8CUUPU", "LR08G9C8"]

    def push_battle(self, battle: Tuple):
        self.db.insert_battle(battle)

    def get_unchecked_player(self, count: int = 1):
        return self.db.get_unchecked_player(count)

    @staticmethod
    def get_players_from_battlelog(battle: Dict[str, Any]):
        players = []
        for team in battle["battle"]["teams"]:
            players.extend((player["tag"][1:], player["name"])
                           for player in team)
        return players

    @staticmethod
    def check_ranked_eligibility(player_stats):
        brawlers = player_stats["brawlers"]
        count = 0
        for brawler in brawlers:
            if brawler["power"] >= 9:
                count += 1
        return count >= 12

    '''def check_ranked_eligibility(brawler):
        trophies = brawler["trophies"]
        if trophies < 18:
            return False
        return True'''

    def get_player_stats(self, player_tags):
        stats = []
        batch_size = 40
        batches = []
        player_count = len(player_tags)
        for i in range(batch_size):
            start = i * player_count // batch_size
            end = (i + 1) * player_count // batch_size
            batches.append(player_tags[start:end])

        threads = [PlayerStatsThread(batch, self.api) for batch in batches]
        for thread in threads:
            thread.start()
            time.sleep(1 / batch_size)
        for thread in threads:
            thread.join()
        for thread in threads:
            stats += thread.result
        return stats

    def get_battlelogs(self, player_tags):
        logs = []
        '''for playerTag in playerTags:
            logs.append(api.getPlayerBattlelog(playerTag))'''
        batch_size = 40
        batches = []
        player_count = len(player_tags)
        for i in range(batch_size):
            start = i * player_count // batch_size
            end = (i + 1) * player_count // batch_size
            batches.append(player_tags[start:end])

        threads = [BattleLogsThread(batch, self.api) for batch in batches]
        for thread in threads:
            thread.start()
            time.sleep(1 / batch_size)
        for thread in threads:
            thread.join()
        for thread in threads:
            logs += thread.result
        return logs

    def check_battle_with_only_eligible_players(self, battle,
                                                eligible_players):
        try:
            for team in battle["battle"]["teams"]:
                for player in team:
                    if player["tag"][1:] in eligible_players:
                        continue
                    else:
                        return False
            return True
        except Exception as e:
            print(e)
            print(json.dumps(battle))
            return False

    def cycle(self):
        batch_size = 1000
        players_list = self.get_unchecked_player(batch_size)
        player_tags = [player[0] for player in players_list]

        start_time = time.time()
        battle_logs = self.get_player_stats(player_tags)
        print(f"Battlelogs took: {time.time() - start_time:.2f} seconds")

        battles = [battle for log in battle_logs for battle in log["items"]]
        new_players = []
        filtered_battles = []

        for battle in battles:  # Get all players from the battles
            if battle["battle"].get("type") == "soloRanked":
                filtered_battles.append(battle)
                new_players.extend(self.get_players_from_battlelog(battle))

        new_players = list(set(new_players))  # Remove duplicates
        player_tags = [player[0] for player in new_players]
        existing_players = self.db.get_if_players_exist(player_tags)
        player_tags = list(set(player_tags) - set(existing_players))

        player_stats_list = self.get_player_stats(player_tags)
        eligible_players = [stats["tag"][1:] for stats in player_stats_list
                            if self.check_ranked_eligibility(stats)]

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print(f"Players: {len(player_stats_list)}")
        print(f"Eligible Players: {len(eligible_players)}")
        print(f"Battles: {len(filtered_battles)}")

        real_battles = []
        for battle in battles:
            if self.check_battle_with_only_eligible_players(battle,
                                                            eligible_players):
                real_battles.append(battle)

        print(f"Real Battles: {len(real_battles)}")

        self.db.insert_many_players(new_players)
        for battle in real_battles:
            battle_id = random.randint(0, 10000000000000)
            unix_time = time.mktime(
                time.strptime(battle["battleTime"], "%Y%m%dT%H%M%S.000Z"))
            unix_time = int(unix_time)
            sql_battle = (battle_id, unix_time, battle["event"]["map"],
                          battle["battle"]["mode"],
                          battle["battle"]["teams"][0][0]["brawler"]["name"],
                          battle["battle"]["teams"][0][1]["brawler"]["name"],
                          battle["battle"]["teams"][0][2]["brawler"]["name"],
                          battle["battle"]["teams"][1][0]["brawler"]["name"],
                          battle["battle"]["teams"][1][1]["brawler"]["name"],
                          battle["battle"]["teams"][1][2]["brawler"]["name"],
                          1 if battle["battle"]["result"] == "victory" else 0)
            self.push_battle(sql_battle)
        self.db.set_many_players_checked(player_tags)
        self.db.commit()

    def feed_db(self):
        print(f"Currently there are {self.db.get_battles_count()} "
              f"battles in the database")
        print(f"Currently there are {self.db.get_players_count()} "
              f"players in the database")

        start_time = time.time()
        if self.db.get_players_count() == 0:
            player_api = self.api.get_player_stats(self.first_player)
            player = (player_api["tag"][1:], player_api["name"])
            self.db.insert_player(player)

        runtime = 1
        while (time.time() - start_time) < 60 * runtime:
            print(f"Time left: "
                  f"{(60 * runtime - (time.time() - start_time)) / 60:.2f} "
                  f"min")
            self.cycle()
            print(f"Checked players: "
                  f"{self.db.get_checked_players_percentage()}")
            print(f"Battles in database: {self.db.get_battles_count()}")
            print(f"Players in database: {self.db.get_players_count()}")


class PlayerStatsThread(threading.Thread):

    def __init__(self, batch: List[str], api: DevBrawlAPI):
        super().__init__()
        self.batch = batch
        self.api = api
        self.result = []

    def run(self):
        for player in self.batch:
            for _ in range(3):
                stats_value = self.api.get_player_stats(player)
                if stats_value is not None:
                    self.result.append(stats_value)
                    break
                time.sleep(0.2)


class BattleLogsThread(threading.Thread):
    def __init__(self, batch: List[str], api: DevBrawlAPI):
        super().__init__()
        self.batch = batch
        self.api = api
        self.result = []

    def run(self):
        for player_tag in self.batch:
            for _ in range(3):
                logs_value = self.api.get_player_battlelog(player_tag)
                if logs_value is not None:
                    self.result.append(logs_value)
                    break
                time.sleep(0.2)


if __name__ == "__main__":
    manager = DevBrawlManager()
    manager.db.reset()
    for player in manager.best_players:
        player_api = manager.api.get_player_stats(player)
        player = (player_api["tag"][1:], player_api["name"])
        manager.db.insert_player(player)

    manager.feed_db()

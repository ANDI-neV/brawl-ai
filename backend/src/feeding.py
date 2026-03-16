from datasource import DevBrawlAPI
from db import Database
import json
import time
import random
from datetime import datetime, timezone
import threading
from typing import Tuple, List, Dict, Any
import argparse
import os

from ranked_utils import (
    get_players_from_ranked_battle,
    is_supported_ranked_battle,
    normalize_brawler_name,
)


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
    def __init__(self, date):
        self.api = DevBrawlAPI()
        self.db = Database()
        random.seed(datetime.now().timestamp())
        self.first_player = "GGCOLO2R"
        self.best_players = [
            "P2G8CUUPU", "LR08G9C8", "GGCOLO2R", "GJPVYUQG",
            "GY80VJP", "PORORCPPQ", "9YPOCYOV", "P2C2J2PGU",
            "202GJJR28", "2OP2GP99", "2UP82YLQ", "PGPUROYG",
            "2OLCGGQVQ", "J99YU9QY", "2RUYYLYVC", "89UGG92L2",
            "PLLRJC2V", "RO29LYYQ", "9GUVOPJ8Y", "9QJGLJJ8",
            "80VJGCLOP", "2CQJG29JL", "Y9C88V9PC", "PGPUROYG",
            "2R9U2C2JP", "9CR2GJGJR", "JYCCCUQC", "9P2CUYJ8P",
            "LLQ8GY8", "J8YPGR", "99YLOPV9V", "LRPUV8J92",
            "8YUVLQQLG", "92VCJYRRC", "8GLUVYQPC",
            "8U92UJGLL", "YOCOPGOPR", "808VYPP", "9C98P9UG2",
            "JUYYJGO", "288LL2VY9",
            "9OVQGVL9G", "82YCOCYC", "QGOGOYPO", "ZULVL8YR9",
            "V998RULC", "YYJQUVR2", "PUG2V98UP", "GJJQPYC9",
            "8YVCLOPPY", "98VOQCOQJ", "GRQLGPU", "P9928Y8U2",
            "#Q88GUGP"
        ]
        time_step = time.mktime(datetime.strptime(date, "%d.%m.%Y").timetuple())
        self.min_timestamp = datetime.fromtimestamp(time_step,
                                                    tz=timezone.utc)

    def push_battle(self, battle: Tuple):
        self.db.insert_battle(battle)

    def get_unchecked_player(self, count: int = 1):
        # Modify this to include more players, not just unchecked ones
        return self.db.get_players_for_expansion(count)

    @staticmethod
    def get_players_from_battlelog(battle: Dict[str, Any]):
        return get_players_from_ranked_battle(battle)

    @staticmethod
    def check_ranked_eligibility(player_stats):
        brawlers = player_stats["brawlers"]
        count = 0
        for brawler in brawlers:
            if brawler["power"] >= 9:
                count += 1
        return count >= 12

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

    @staticmethod
    def parse_iso_timestamp(timestamp_str):
        return (datetime.strptime(timestamp_str,
                                 "%Y%m%dT%H%M%S.%fZ")
                .replace(tzinfo=timezone.utc))

    @staticmethod
    def is_supported_ranked_battle(battle: Dict[str, Any]) -> bool:
        return is_supported_ranked_battle(battle)

    @staticmethod
    def normalize_brawler_name(name: str) -> str:
        return normalize_brawler_name(name)

    def to_sql_battle(self, battle: Dict[str, Any]) -> Tuple:
        battle_id = random.randint(0, 10000000000000)
        unix_time = int(
            time.mktime(time.strptime(battle["battleTime"], "%Y%m%dT%H%M%S.000Z"))
        )
        teams = battle["battle"]["teams"]
        return (
            battle_id,
            unix_time,
            battle["event"]["map"],
            battle["battle"]["mode"],
            self.normalize_brawler_name(teams[0][0]["brawler"]["name"]),
            self.normalize_brawler_name(teams[0][1]["brawler"]["name"]),
            self.normalize_brawler_name(teams[0][2]["brawler"]["name"]),
            self.normalize_brawler_name(teams[1][0]["brawler"]["name"]),
            self.normalize_brawler_name(teams[1][1]["brawler"]["name"]),
            self.normalize_brawler_name(teams[1][2]["brawler"]["name"]),
            1 if battle["battle"]["result"] == "victory" else 0,
        )

    def get_battlelogs(self, player_tags):
        logs = []
        player_tags_to_remove = []
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
            player_tags_to_remove += thread.player_tags_to_remove
        return logs, player_tags_to_remove

    @staticmethod
    def check_battle_with_only_eligible_players(battle,
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
        try:
            batch_size = 200
            players_list = self.get_unchecked_player(batch_size)
            player_tags = [player[0] for player in players_list]

            start_time = time.time()
            battle_logs, player_tags_to_remove = self.get_battlelogs(player_tags)
            for player_tag in player_tags_to_remove:
                self.db.delete_player(player_tag)

            print(f"Battlelogs took: {time.time() - start_time:.2f} seconds")
            battles = [battle for log in battle_logs for battle in log["items"]]
            new_players = []
            filtered_battles = []

            for battle in battles:
                if (
                    self.is_supported_ranked_battle(battle)
                    and self.parse_iso_timestamp(battle["battleTime"]) >= self.min_timestamp
                ):
                    filtered_battles.append(battle)
                    new_players.extend(self.get_players_from_battlelog(battle))

            new_players = list(set(new_players))  # Remove duplicates
            new_player_tags = [player[0] for player in new_players]
            existing_players = self.db.get_if_players_exist(new_player_tags)
            existing_players = set(existing_players)
            new_players = [player for player in new_players if player[0] not in existing_players]

            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            print(f"New players: {len(new_players)}")
            print(f"Battles: {len(filtered_battles)}")

            self.db.insert_many_players(new_players)
            for battle in filtered_battles:
                self.push_battle(self.to_sql_battle(battle))
            self.db.set_many_players_checked(player_tags)
            self.db.commit()

        except Exception as e:
            print(f"Error in cycle: {str(e)}")
            self.db.rollback()

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

        runtime = 120
        while (time.time() - start_time) < 60 * runtime and self.db.get_checked_players_percentage() < 0.98 and self.db.get_battles_count() <= 275000:
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
        self.player_tags_to_remove = []

    def run(self):
        for player_tag in self.batch:
            for _ in range(3):
                logs_value = self.api.get_player_battlelog(player_tag)
                if logs_value == "notFound":
                    self.player_tags_to_remove.append(player_tag)
                    break
                elif logs_value is not None:
                    self.result.append(logs_value)
                    break
                else:
                    time.sleep(0.2)


def set_last_update(date):
    with open(os.path.join("./out", 'last_update.json'), 'w') as f:
        json.dump(date, f)


def get_last_update():
    with open(os.path.join("./out", 'last_update.json'), 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brawl Stars Data Collection')
    parser.add_argument('--last_update',
                        type=str,
                        help='Date for data collection (format: DD.MM.YYYY)')

    args = parser.parse_args()
    date = ""
    print(args)
    if args.last_update:
        set_last_update(args.last_update)
        date = args.last_update
        print(date)
    else:
        date = get_last_update()

    manager = DevBrawlManager(date)
    if manager.db.get_players_count() == 0:
        for player in manager.best_players:
            player_stats = manager.api.get_player_stats(player)
            if player_stats:
                player = (player_stats["tag"][1:], player_stats["name"])
                manager.db.insert_player(player)

    manager.feed_db()

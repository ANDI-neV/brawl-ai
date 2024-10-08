import configparser
import time
from typing import List, Tuple

import psycopg2
from psycopg2 import pool


class ConnectionPoolManager:
    _instance = None
    _pool = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def initpool(cls):
        try:
            config = configparser.ConfigParser()
            config.read('config.ini')
            cls._pool = psycopg2.pool.ThreadedConnectionPool(
                1, 1000,
                host=config['Credentials']['host'],
                port=5432,
                database=config['Credentials']['database'],
                user=config['Credentials']['username'],
                password=config['Credentials']['password'],
            )
        except Exception as e:
            print(f"Error: {e}")
            raise e

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls.initpool()
        return cls._instance

    @classmethod
    def getconn(cls):
        return cls._pool.getconn()

    @classmethod
    def putconn(cls, conn):
        cls._pool.putconn(conn)

    def __del__(cls):
        if cls._pool is not None:
            cls._pool.closeall()


class Database:
    def __init__(self):
        manager = ConnectionPoolManager.instance()
        self.conn = manager.getconn()
        self.cur = self.conn.cursor()

    def set_all_players_can_ranked(self):
        self.cur.execute("UPDATE players SET canRanked=-1")  # -1 means not checked yet
        self.conn.commit()

    def commit(self):
        self.conn.commit()

    def delete_all_battles(self):
        self.cur.execute(
            "TRUNCATE TABLE battles"
        )
        db.commit()

    def delete_all_players(self):
        self.cur.execute(
            "TRUNCATE TABLE players"
        )
        db.commit()

    def get_unchecked_player(self, count: int = 1) -> List[Tuple]:
        self.cur.execute(
            "SELECT * FROM players WHERE checked = 0 AND canRanked=-1 LIMIT %s",
            (count,)
        )
        return self.cur.fetchall()

    def get_unknown_ranked_player(self, count: int = 1) -> List[Tuple]:
        self.cur.execute(
            "SELECT * FROM players WHERE canRanked = -1 LIMIT %s",
            (count,)
        )
        return self.cur.fetchall()

    def get_battles_count(self) -> int:
        self.cur.execute("SELECT COUNT(*) FROM battles")
        return self.cur.fetchone()[0]

    def get_players_count(self) -> int:
        self.cur.execute("SELECT COUNT(*) FROM players")
        return self.cur.fetchone()[0]

    def get_players_for_expansion(self, count: int = 1000) -> List[Tuple]:
        self.cur.execute(
            """
            SELECT * FROM players 
            ORDER BY RANDOM()
            LIMIT %s
            """,
            (count,)
        )
        return self.cur.fetchall()

    def update_player_checked_status(self, tag: str):
        self.cur.execute(
            """
            UPDATE players 
            SET checked = 1, 
                last_checked = CURRENT_TIMESTAMP
            WHERE tag = %s
            """,
            (tag,)
        )
        self.conn.commit()

    def get_checked_players_percentage(self) -> float:
        self.cur.execute("SELECT COUNT(*) FROM players WHERE checked=1")
        checked = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM players")
        total = self.cur.fetchone()[0]
        return checked / total

    def insert_battle(self, battle: Tuple):
        # check by BattleTime
        self.cur.execute("SELECT * FROM battles WHERE battleTime=%s", (battle[1],))
        if self.cur.fetchone():
            return
        self.cur.execute(
            """INSERT INTO battles
            (id, battleTime, map, mode, a1, a2, a3, b1, b2, b3, result)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            battle
        )
        self.commit()

    def insert_player(self, player: Tuple):
        tag, name = player
        self.cur.execute("SELECT * FROM players WHERE tag=%s", (tag,))
        if self.cur.fetchone():
            return
        self.cur.execute(
            "INSERT INTO players (tag, name, checked, canRanked) VALUES (%s, %s, %s, %s)",
            (tag, name, 0, -1)
        )
        self.commit()

    def print_all_battles(self):
        self.cur.execute("SELECT * FROM battles")
        print(self.cur.fetchall())

    def get_if_many_players_exist(self, tags: List[str]) -> List[Tuple]:
        self.cur.execute("SELECT tag FROM players WHERE tag = ANY(%s)", [tags])
        return self.cur.fetchall()

    def insert_many_players(self, players: List[Tuple]):
        # First, filter out the players that already exist
        tags = [player[0] for player in players]
        existing = self.get_if_many_players_exist(tags)
        existing_tags = set(player[0] for player in existing)

        print(f"Attempting to insert {len(players)} players")
        u_players = [
            player for player in players
            if player[0] not in existing_tags
        ]
        print(f"After filtering out existing players, {len(u_players)} players remain")

        if not u_players:
            print("No new players to insert.")
            return

        # Prepare the SQL query
        insert_query = """
        INSERT INTO players (tag, name, checked, canRanked)
        VALUES (%s, %s, 0, -1)
        ON CONFLICT (tag) DO UPDATE
        SET name = EXCLUDED.name,
            checked = 0,
            canRanked = -1
        """

        # Execute the query for each player
        for player in u_players:
            try:
                self.cur.execute(insert_query, (player[0], player[1]))
            except Exception as e:
                print(f"Error inserting player {player[0]}: {str(e)}")
                continue

        self.conn.commit()
        print(f"Successfully inserted/updated {len(u_players)} players")

    def set_player_checked(self, tag: str):
        self.cur.execute("UPDATE players SET checked=1 WHERE tag=%s", (tag,))
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def set_many_players_checked(self, tags):
        self.cur.execute("UPDATE players SET checked=1 WHERE tag = ANY(%s)", [tags])
        self.conn.commit()

    def get_winrate(self, brawler: str) -> float:
        self.cur.execute(
            """SELECT COUNT(*) FROM battles
            WHERE (a1=%s OR a2=%s OR a3=%s) AND result=1""",
            (brawler, brawler, brawler)
        )
        wins = self.cur.fetchone()[0]
        self.cur.execute(
            """SELECT COUNT(*) FROM battles
            WHERE (b1=%s OR b2=%s OR b3=%s) AND result=0""",
            (brawler, brawler, brawler)
        )
        wins += self.cur.fetchone()[0]
        self.cur.execute(
            "SELECT COUNT(*) FROM battles WHERE (a1=%s OR a2=%s OR a3=%s)",
            (brawler, brawler, brawler)
        )
        total = self.cur.fetchone()[0]
        self.cur.execute(
            "SELECT COUNT(*) FROM battles WHERE (b1=%s OR b2=%s OR b3=%s)",
            (brawler, brawler, brawler)
        )
        total += self.cur.fetchone()[0]
        return wins / total if total > 0 else 0

    def get_all_brawlers(self) -> List[str]:
        brawlers = []
        for column in ['a1', 'a2', 'a3', 'b1', 'b2', 'b3']:
            self.cur.execute(f"SELECT DISTINCT {column} FROM battles")
            brawlers.extend([b[0] for b in self.cur.fetchall()])
        return list(dict.fromkeys(brawlers))

    def get_unique_match_combos(self) -> List[Tuple]:
        self.cur.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT a1, a2, a3, b1, b2, b3 FROM battles)"
        )
        return self.cur.fetchall()

    def get_unique_battles_with_more_than_x_matches(self, x: int) -> List[Tuple]:
        self.cur.execute(
            """SELECT * FROM
            (SELECT a1, a2, a3, b1, b2, b3 FROM battles
            GROUP BY a1, a2, a3, b1, b2, b3 HAVING COUNT(*) > %s)""",
            (x,)
        )
        return self.cur.fetchall()

    def reset(self):
        self.cur.execute("DELETE FROM battles")
        self.cur.execute("DELETE FROM players")
        self.conn.commit()

    def get_if_players_exist(self, tags: List[str]) -> List[Tuple]:
        self.cur.execute("SELECT tag FROM players WHERE tag = ANY(%s)", [tags])
        return self.cur.fetchall()

    def delete_old_battles(self):
        current_unix = int(time.time())
        days = current_unix - 86400 * 60  # 60 days
        self.cur.execute("DELETE FROM battles WHERE battleTime < %s", (days,))
        self.conn.commit()

    def get_map_winrate(self, map_name: str) -> float:
        self.cur.execute(
            "SELECT COUNT(*) FROM battles WHERE map=%s AND result=1",
            (map_name,)
        )
        wins = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE map=%s", (map_name,))
        total = self.cur.fetchone()[0]
        return wins / total if total > 0 else 0

    def get_all_maps(self) -> List[Tuple]:
        self.cur.execute("SELECT DISTINCT map FROM battles")
        return self.cur.fetchall()

    def check_brawler_significance_for_map(
        self, brawler: str, map_name: str = None
    ) -> float:
        query = """
            SELECT
                SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1)
                    OR ((b1=%s OR b2=%s OR b3=%s)) THEN 1 ELSE 0 END) as matches,
                COUNT(*) as total_games
            FROM battles
        """
        params = (brawler,) * 6

        if map_name:
            query += " WHERE map=%s"
            params += (map_name,)

        self.cur.execute(query, params)
        matches, total_games = self.cur.fetchone()

        return matches / total_games if total_games > 0 else 0

    def check_brawler_winrate_for_map(
        self, brawler: str, map_name: str = None
    ) -> float:
        query = """
            SELECT
                SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1)
                    OR ((b1=%s OR b2=%s OR b3=%s) AND result=0) THEN 1 ELSE 0 END) as wins,
                COUNT(*) as total_games
            FROM battles
            WHERE a1=%s OR a2=%s OR a3=%s OR b1=%s OR b2=%s OR b3=%s
        """
        params = (brawler,) * 12

        if map_name:
            query += " AND map=%s"
            params += (map_name,)

        self.cur.execute(query, params)
        wins, total_games = self.cur.fetchone()

        return wins / total_games if total_games > 0 else 0


class ThreadedDBWorker:
    def __init__(self):
        self.conn = ConnectionPoolManager.instance().getconn()
        self.cur = self.conn.cursor()

    def __del__(self):
        ConnectionPoolManager.instance().putconn(self.conn)

if __name__ == "__main__":
    db = Database()
    print(db.get_battles_count())
import psycopg2
from psycopg2 import pool
import random
import time
import os
import configparser

'''psycopg2.connect(
    host=os.getenv("REDACTED"),
    port=5432,
    database="REDACTED",
    user="REDACTED",
    password="REDACTED"
)'''

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
            print("Error: " + str(e))
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

    def setAllPlayersCanRanked(self):
        self.cur.execute("UPDATE players SET canRanked=-1") # -1 means not checked yet
        self.conn.commit()

    def commit(self):
        self.conn.commit()

    def get_unchecked_player(self, count=1):
        self.cur.execute("SELECT * FROM players WHERE checked = 0 AND canRanked=-1 LIMIT %s", (count,))
        return self.cur.fetchall()
    
    def get_unknown_ranked_player(self, count=1):
        self.cur.execute("SELECT * FROM players WHERE canRanked = -1 LIMIT %s", (count,))
        return self.cur.fetchall()
    
    def get_battles_count(self):
        self.cur.execute("SELECT COUNT(*) FROM battles")
        return self.cur.fetchone()[0]
    
    def get_players_count(self):
        self.cur.execute("SELECT COUNT(*) FROM players")
        return self.cur.fetchone()[0]
    
    def get_checked_players_percentage(self):
        self.cur.execute("SELECT COUNT(*) FROM players WHERE checked=1")
        checked = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM players")
        total = self.cur.fetchone()[0]
        return checked / total
    
    def insert_battle(self, battle):
    # check by BattleTime
        self.cur.execute("SELECT * FROM battles WHERE battleTime=%s", (battle[1],))
        if self.cur.fetchone():
            return
        self.cur.execute("INSERT INTO battles (id, battleTime, map, mode, a1, a2, a3, b1, b2, b3, result) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", battle)

    def insert_player(self, player):
        tag = player[0]
        name = player[1]
        self.cur.execute("SELECT * FROM players WHERE tag=%s", (tag,))
        if self.cur.fetchone():
            return
        self.cur.execute("INSERT INTO players (tag, name, checked, canRanked) VALUES (%s, %s, %s, %s)", (tag, name, 0, -1))

    def get_if_many_players_exist(self, tags):
        self.cur.execute("SELECT tag FROM players WHERE tag = ANY(%s)", [tags])
        return self.cur.fetchall()

    def insert_many_players(self, players):
        # first filter out the players that already exist
        tags = [player[0] for player in players]
        existing = self.get_if_many_players_exist(tags)
        print("Currently " + str(len(players)) + " players exist")
        u_players = []
        # players are (tag,name)
        # existing is (tag,)
        for player in players:
            if player[0] not in [x[0] for x in existing]:
                u_players.append(player)
        print("After filtering out existing players, " + str(len(u_players)) + " players remain")
        sqlplayers = [(player[0], player[1], 0, -1) for player in u_players]
        args_str = ','.join(self.cur.mogrify("(%s,%s,%s,%s)", x).decode('utf-8') for x in sqlplayers)
        self.cur.execute("INSERT INTO players (tag, name, checked, canRanked) VALUES " + args_str)
        

    def set_player_checked(self, tag):
        self.cur.execute("UPDATE players SET checked=1 WHERE tag=%s", (tag,))
        self.conn.commit()

    def set_many_players_checked(self, tags):
        self.cur.execute("UPDATE players SET checked=1 WHERE tag = ANY(%s)", [tags])
        self.conn.commit()

    def getWinrate(self, brawler):
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (a1=%s OR a2=%s OR a3=%s) AND result=1", (brawler, brawler, brawler))
        wins = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (b1=%s OR b2=%s OR b3=%s) AND result=0", (brawler, brawler, brawler))
        wins += self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (a1=%s OR a2=%s OR a3=%s)", (brawler, brawler, brawler))
        total = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (b1=%s OR b2=%s OR b3=%s)", (brawler, brawler, brawler))
        total += self.cur.fetchone()[0]
        return wins / total if total > 0 else 0
    
    def getAllBrawlers(self):
        self.cur.execute("SELECT DISTINCT a1 FROM battles")
        a1 = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT a2 FROM battles")
        a2 = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT a3 FROM battles")
        a3 = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT b1 FROM battles")
        b1 = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT b2 FROM battles")
        b2 = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT b3 FROM battles")
        b3 = self.cur.fetchall()
        brawlers = []
        for brawler in a1 + a2 + a3 + b1 + b2 + b3:
            brawlers.append(brawler[0])
        # remove duplicates
        brawlers = list(dict.fromkeys(brawlers))
        return brawlers


    def getUniqueMatchCombos(self):# select battles where a1, a2, a3, b1, b2, b3 are unique, then count
        self.cur.execute("SELECT COUNT(*) FROM (SELECT DISTINCT a1, a2, a3, b1, b2, b3 FROM battles)")
        return self.cur.fetchall()

    def getUniqueBattlesWithMoreThanXMatches(self, x):
        self.cur.execute("SELECT * FROM (SELECT a1, a2, a3, b1, b2, b3 FROM battles GROUP BY a1, a2, a3, b1, b2, b3 HAVING COUNT(*) > %s)", (x,))
        return self.cur.fetchall()
    
    def reset(self):
        self.cur.execute("DELETE FROM battles")
        self.cur.execute("DELETE FROM players")
        self.conn.commit()

    def get_if_players_exist(self, tags):
        self.cur.execute("SELECT tag FROM players WHERE tag = ANY(%s)", [tags])
        return self.cur.fetchall()

    def deleteOldBattles(self):
        currentunix = int(time.time())
        days = currentunix - 86400 * 60 # 60 days
        self.cur.execute("DELETE FROM battles WHERE battleTime < %s", (days,))
        self.conn.commit()

    def getMapWinrate(self, map):
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE map=%s AND result=1", (map,))
        wins = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE map=%s", (map,))
        total = self.cur.fetchone()[0]
        return wins / total if total > 0 else 0


    def getAllMaps(self):
        self.cur.execute("SELECT DISTINCT map FROM battles")
        return self.cur.fetchall()

    def checkBrawlerSignificanceForMap(self, brawler, map=None):
        if map is None:
            self.cur.execute("""
                SELECT 
                    SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1) OR ((b1=%s OR b2=%s OR b3=%s)) THEN 1 ELSE 0 END) as matches,
                    COUNT(*) as total_games
                FROM battles 
            """, (brawler,) * 6)
        else:
            self.cur.execute("""
                SELECT 
                    SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1) OR ((b1=%s OR b2=%s OR b3=%s)) THEN 1 ELSE 0 END) as matches,
                    COUNT(*) as total_games
                FROM battles 
                WHERE map=%s
            """, (brawler,) * 6 + (map,))

        result = self.cur.fetchone()
        matches, total_games = result[0], result[1]

        if total_games == 0:
            return 0
        return matches / total_games

    def checkBrawlerWinrateForMap(self, brawler, map=None):
        if map is None:
            self.cur.execute("""
                SELECT 
                    SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1) OR ((b1=%s OR b2=%s OR b3=%s) AND result=0) THEN 1 ELSE 0 END) as wins,
                    COUNT(*) as total_games
                FROM battles 
                WHERE a1=%s OR a2=%s OR a3=%s OR b1=%s OR b2=%s OR b3=%s
            """, (brawler,) * 12)
        else:
            self.cur.execute("""
                SELECT 
                    SUM(CASE WHEN ((a1=%s OR a2=%s OR a3=%s) AND result=1) OR ((b1=%s OR b2=%s OR b3=%s) AND result=0) THEN 1 ELSE 0 END) as wins,
                    COUNT(*) as total_games
                FROM battles 
                WHERE (a1=%s OR a2=%s OR a3=%s OR b1=%s OR b2=%s OR b3=%s) AND map=%s
            """, (brawler,) * 12 + (map,))

        result = self.cur.fetchone()
        wins, total_games = result[0], result[1]

        if total_games == 0:
            return 0
        return wins / total_games
    

class ThreadedDBWorker:
    def __init__(self):
        self.conn = ConnectionPoolManager.instance().getconn()
        self.cur = self.conn.cursor()

    def __del__(self):
        ConnectionPoolManager.instance().putconn(self.conn)


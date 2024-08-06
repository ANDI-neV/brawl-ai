import sqlite3
import random
import time
import os


class Database:
    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.conn = sqlite3.connect(os.path.join(here, 'out/db/games.db'))
        self.cur = self.conn.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS battles (id INTEGER PRIMARY KEY, battleTime INTEGER, map TEXT, mode TEXT, a1 TEXT, a2 TEXT, a3 TEXT, b1 TEXT, b2 TEXT, b3 TEXT, result INTEGER)")
        self.cur.execute("CREATE TABLE IF NOT EXISTS players (tag TEXT PRIMARY KEY, name TEXT, checked INTEGER)")
        self.conn.commit()

    def commit(self):
        self.conn.commit()

    def get_unchecked_player(self):
        self.cur.execute("SELECT * FROM players WHERE checked=0 LIMIT 1")
        return self.cur.fetchone()
    
    def get_battles_count(self):
        self.cur.execute("SELECT COUNT(*) FROM battles")
        return self.cur.fetchone()[0]
    
    def get_players_count(self):
        self.cur.execute("SELECT COUNT(*) FROM players")
        return self.cur.fetchone()[0]
    
    def insert_battle(self, battle):
        # check by BattleTime
        if self.cur.execute("SELECT * FROM battles WHERE battleTime=?", (battle[1],)).fetchone():
            return
        self.cur.execute("INSERT INTO battles (id, battleTime, map, mode, a1, a2, a3, b1, b2, b3, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", battle)

    def insert_player(self, player):
        tag = player[0]
        name = player[1]
        if self.cur.execute("SELECT * FROM players WHERE tag=?", (tag,)).fetchone():
            return
        self.cur.execute("INSERT INTO players (tag, name, checked) VALUES (?, ?, ?)", (tag, name, 0))

    def set_player_checked(self, tag):
        self.cur.execute("UPDATE players SET checked=1 WHERE tag=?", (tag,))
        self.conn.commit()

    def getWinrate(self, brawler):
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (a1=? OR a2=? OR a3=?) AND result=1", (brawler, brawler, brawler))
        wins = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (b1=? OR b2=? OR b3=?) AND result=0", (brawler, brawler, brawler))
        wins += self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (a1=? OR a2=? OR a3=?)", (brawler, brawler, brawler))
        total = self.cur.fetchone()[0]
        self.cur.execute("SELECT COUNT(*) FROM battles WHERE (b1=? OR b2=? OR b3=?)", (brawler, brawler, brawler))
        total += self.cur.fetchone()[0]
        return wins / total
    
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
        self.cur.execute("SELECT * FROM (SELECT a1, a2, a3, b1, b2, b3 FROM battles GROUP BY a1, a2, a3, b1, b2, b3 HAVING COUNT(*) > ?)", (x,))
        return self.cur.fetchall()
    
    def reset(self):
        self.cur.execute("DELETE FROM battles")
        self.cur.execute("UPDATE players SET checked=0")
        self.conn.commit()

    def deleteOldBattles(self):
        currentunix = int(time.time())
        days = currentunix - 86400 * 60 # 60 days
        self.cur.execute("DELETE FROM battles WHERE battleTime < ?", (days,))
        self.conn.commit()

    def getAllMaps(self):
        self.cur.execute("SELECT DISTINCT map FROM battles")
        return self.cur.fetchall()
    
    def checkBrawlerSignificance(self, brawler, map=None):
        if map == None:
            self.cur.execute("SELECT COUNT(*) FROM battles WHERE a1=? OR a2=? OR a3=? OR b1=? OR b2=? OR b3=?", (brawler, brawler, brawler, brawler, brawler, brawler))
        else:
            self.cur.execute("SELECT COUNT(*) FROM battles WHERE (a1=? OR a2=? OR a3=? OR b1=? OR b2=? OR b3=?) AND map=?", (brawler, brawler, brawler, brawler, brawler, brawler, map))
        return self.cur.fetchone()[0]
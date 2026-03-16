CREATE TABLE IF NOT EXISTS players (
    tag TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    last_checked TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT TIMESTAMP '0001-01-01 00:00:00'
);

CREATE TABLE IF NOT EXISTS battles (
    id BIGINT PRIMARY KEY,
    battleTime BIGINT NOT NULL,
    map TEXT NOT NULL,
    mode TEXT NOT NULL,
    a1 TEXT NOT NULL,
    a2 TEXT NOT NULL,
    a3 TEXT NOT NULL,
    b1 TEXT NOT NULL,
    b2 TEXT NOT NULL,
    b3 TEXT NOT NULL,
    result SMALLINT NOT NULL CHECK (result IN (0, 1))
);

CREATE INDEX IF NOT EXISTS battles_battletime_idx ON battles (battleTime);
CREATE INDEX IF NOT EXISTS battles_map_idx ON battles (map);
CREATE INDEX IF NOT EXISTS battles_mode_idx ON battles (mode);
CREATE INDEX IF NOT EXISTS players_last_checked_idx ON players (last_checked);

CREATE UNIQUE INDEX IF NOT EXISTS battles_dedupe_idx
ON battles (battleTime, map, mode, a1, a2, a3, b1, b2, b3, result);

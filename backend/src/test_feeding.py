import unittest

from ranked_utils import get_players_from_ranked_battle, is_supported_ranked_battle


def _battle(battle_type: str):
    return {
        "battleTime": "20260316T003010.000Z",
        "event": {"map": "Ring of Fire"},
        "battle": {
            "type": battle_type,
            "mode": "hotZone",
            "result": "victory",
            "teams": [
                [
                    {"tag": "#AAA", "name": "One", "brawler": {"name": "COLT"}},
                    {"tag": "#BBB", "name": "Two", "brawler": {"name": "BELLE"}},
                    {"tag": "#CCC", "name": "Three", "brawler": {"name": "FINX"}},
                ],
                [
                    {"tag": "#DDD", "name": "Four", "brawler": {"name": "EMZ"}},
                    {"tag": "#EEE", "name": "Five", "brawler": {"name": "BEA"}},
                    {"tag": "#FFF", "name": "Six", "brawler": {"name": "GLOWBERT"}},
                ],
            ],
        },
    }


class FeedingBattleTypeTests(unittest.TestCase):
    def test_ranked_is_supported(self):
        self.assertTrue(is_supported_ranked_battle(_battle("ranked")))

    def test_solo_ranked_is_supported(self):
        self.assertTrue(is_supported_ranked_battle(_battle("soloRanked")))

    def test_tournament_is_not_supported(self):
        self.assertFalse(is_supported_ranked_battle(_battle("tournament")))

    def test_player_extraction_no_longer_depends_on_trophies(self):
        players = get_players_from_ranked_battle(_battle("ranked"))
        self.assertEqual(len(players), 6)
        self.assertIn(("AAA", "One"), players)


if __name__ == "__main__":
    unittest.main()

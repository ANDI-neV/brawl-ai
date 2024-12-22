import logging
from datetime import datetime
import subprocess
from db import Database
import argparse

class BrawlAIPipeline:
    def __init__(self, rank_threshold: int, initial_player_id: str, last_update: str):
        self.rank_threshold = rank_threshold
        self.initial_player_id = initial_player_id
        self.last_update = last_update
        self.db = Database()

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'./out/logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ],
        )
        self.logger = logging.getLogger(__name__)

    def update_brawler_data(self) -> None:
        try:
            subprocess.run(["python", "scraper.py"], check=True)
            self.logger.info("Brawler data updated successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Web scraping failed: {str(e)}")
            raise

    def feed_database(self) -> None:
        try:
            subprocess.run(["python", "feeding.py", "--last_update", self.last_update], check=True)
            self.logger.info("Database feeding completed")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Feeding database failed: {str(e)}")
            raise

    def run_pipeline(self):
        try:
            self.logger.info("Starting pipeline execution")

            self.db.delete_all_battles()
            #self.update_brawler_data()
            self.feed_database()
            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def validate_date(date_string):
    try:
        datetime.strptime(date_string, "%d.%m.%Y")
        return True
    except ValueError:
        return False


class NoArgumentsError(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brawl Stars Data Collection')
    parser.add_argument('--last_update',
                        type=str,
                        help='Date for data collection (format: DD.MM.YYYY)',
                        required=True)

    args = parser.parse_args()
    date = ""
    if args:
        date = args.last_update

    pipeline = BrawlAIPipeline(
        rank_threshold=25000,
        initial_player_id="INITIAL_PLAYER_TAG",
        last_update=date
    )

    try:
        success = pipeline.run_pipeline()
        if success:
            print("Pipeline executed successfully!")
        else:
            print("Pipeline completed but validation failed")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")

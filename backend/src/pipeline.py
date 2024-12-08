import logging
import os
from datetime import datetime
import psycopg2
from typing import Optional
import subprocess
import time
from db import Database

class BrawlAIPipeline:
    def __init__(self, rank_threshold: int, initial_player_id: str, last_update: str):
        self.rank_threshold = rank_threshold
        self.initial_player_id = initial_player_id

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def update_brawler_data(self) -> None:
        try:
            subprocess.run(["python", "web_scraper.py"], check=True)
            self.logger.info("Brawler data updated successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Web scraping failed: {str(e)}")
            raise

    def retrain_model(self) -> None:
        try:
            subprocess.run(["python", "train_model.py"], check=True)
            self.logger.info("Model retraining completed")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Model retraining failed: {str(e)}")
            raise

    def run_pipeline(self) -> bool:
        try:
            self.logger.info("Starting pipeline execution")

            db = Database()

            db.delete_all_battles()

            is_valid = self.validate_model()

            if is_valid:
                self.logger.info("Pipeline completed successfully")
            else:
                self.logger.warning("Pipeline completed but model validation failed")

            return is_valid

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def validate_date(date_string):
    try:
        datetime.strptime(date_string, "%d.%m.%Y")
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    print("Enter the date of the last update in the form dd.mm.yyyy:")

    date = input()
    if not validate_date(date):
        print("Invalid date. Please use format dd.mm.yyyy")

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

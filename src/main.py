from ai.datasource import BrawlStarsAPI

if __name__ == "__main__":
    api = BrawlStarsAPI()
    print(api.get_data())


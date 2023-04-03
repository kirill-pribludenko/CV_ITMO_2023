import os

from dotenv import load_dotenv
from sentinelsat import SentinelAPI

# User & Password you can get on https://scihub.copernicus.eu/
USER = os.getenv("user")
PASSWORD = os.getenv("password")
LINK = "https://apihub.copernicus.eu/apihub"
# Product id it is indetifier of aoi
PRODUCT_ID = "62a25193-0ea7-46a7-9ef7-970bd0040f13"


def main():
    load_dotenv()
    api = SentinelAPI(USER, PASSWORD, LINK, show_progressbars=True)
    api.download(PRODUCT_ID, directory_path="./raw")


if __name__ == "__main__":
    main()

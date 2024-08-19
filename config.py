import os
from pathlib import Path
from pprint import pprint

PATH_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATH_PROJECT_IMAGES = os.path.join(PATH_PROJECT_ROOT, "images")

PATH_BACKEND_DIR = Path(__file__).parent
PATH_DAO_DIR = str(PATH_BACKEND_DIR / "dao")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{PATH_DAO_DIR}/hope.db"

if __name__ == "__main__":
    print(PATH_PROJECT_ROOT)
    pprint(PATH_BACKEND_DIR)
    pprint(SQLALCHEMY_DATABASE_URL)
    # pprint(str(PATH_BACKEND_DIR))
    # pprint(SQLALCHEMY_DB_NAME)

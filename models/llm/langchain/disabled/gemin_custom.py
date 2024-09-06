import os

import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MY_BASE_URL = "www.openaiee.icehopeflow.com"
GEMINI_API_URL = f'https://{MY_BASE_URL}/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}'


def query():
    response = requests.get(GEMINI_API_URL)
    print(response)


if __name__ == '__main__':
    query()

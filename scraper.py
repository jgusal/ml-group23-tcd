from time import sleep
import requests
import datetime

# Parameters
URL = 'https://api.jcdecaux.com/vls/v1/stations?apiKey=a9e2b6162328756a94ef2d8af7e6923efdc0b457'
INTERVAL = 15 * 60

while True:
    request = requests.get(URL)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('./json/%s' % timestamp, 'w') as out:
        out.write(request.text)
    sleep(INTERVAL)

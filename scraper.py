from time import sleep
import requests
import datetime
import json

# Parameters
URL = 'https://api.jcdecaux.com/vls/v1/stations?apiKey=a9e2b6162328756a94ef2d8af7e6923efdc0b457'
INTERVAL = 15 * 60

while True:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    try:    
        request = requests.get(URL)
        request.raise_for_status()
        with open('./json/%s.json' % timestamp, 'w') as out:
            data = {"timestamp": timestamp, "payload": request.json()}
            out.write(json.dumps(data))
    except Exception as ex:
        with open('./error/%s' % timestamp, 'w') as out:
            out.write(str(ex))
            
    sleep(INTERVAL)

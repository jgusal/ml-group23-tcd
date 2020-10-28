from time import sleep
import requests
import datetime
import json
import datetime
from time import time

geo_data = json.load(open('geocoords.json'))
URL = "http://api.openweathermap.org/data/2.5/onecall/timemachine?lat=%s&lon=%s&dt=%s&appid=1cc7c30d80001210a8b9642bb71292fe"

DAILY_INTERVAL = 24 * 60 * 60
REQUEST_INTERVAL = 60

while True:
    request_number = 0
    for i in geo_data:
        req_timestamp = int((
            datetime.datetime.fromtimestamp(time()) - datetime.timedelta(days=5)
        ).timestamp())
        req_params = (i["position"]['lat'], i["position"]['lng'], req_timestamp)
        req_url = URL % req_params

        try:    
            request = requests.get(req_url)
            request.raise_for_status()
            with open('./json/%s_%s_%s.json' % (i["position"]['lat'], i["position"]['lng'], req_timestamp), 'w') as out:
                json.dump(request.json(), out)
        except Exception as ex:
            with open('./error/%s_%s_%s.json' % (i["position"]['lat'], i["position"]['lng'], req_timestamp), 'w') as out:
                out.write(str(ex))
        if request_number % 59 == 0 and request_number != 0:
            request_number = 0
            sleep(REQUEST_INTERVAL)
    sleep(DAILY_INTERVAL)

# x = {}
# for i in geodata_dump:
#     j = str(i['position']['lat'])+str(i['position']['lng'])
#     if j not in x:
#         x[j] = 0
#     x[j] += 1

# URL = 'http://api.openweathermap.org/data/2.5/onecall/timemachine?lat=60.99&lon=30.9&dt=1603905114&appid=1cc7c30d80001210a8b9642bb71292fe'

# request = requests.get(URL)
# # data = {"timestamp": timestamp, "payload": request.json()}
# # out.write(json.dumps(data))
# print(request.json())
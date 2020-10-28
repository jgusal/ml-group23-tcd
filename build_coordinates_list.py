from time import sleep
import requests
import datetime
import json
from haversine import haversine, Unit

geo_coords = json.load(open("./bike_locations.json"))

cities = {}
for i in geo_coords:
    city = i['contract_name']
    if city not in cities:
        cities[city] = []
    cities[city].append(i)

geodata_dump = []
for k, v in cities.items():
    for i in range(len(v)):
        for j in range(i, len(v)):
            i_coor = (v[i]['position']['lat'], v[i]['position']['lng'])
            j_coor = (v[j]['position']['lat'], v[j]['position']['lng'])
            if i == j:
                continue
            if haversine(i_coor, j_coor) > 8.17:
                geodata_dump.append(v[i])
                break

print(len(geodata_dump))

json.dump(geodata_dump, open("./geocoords.json", "w"))
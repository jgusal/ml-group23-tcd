import csv
import json
import os
import shutil

input_data_directory = "data/ml_scraper/weather"
output_data = "data/select_data/weather.csv"
try:
    os.remove(output_data)
except:
    pass
output_data_file = open(output_data, "w+")
data_files = os.listdir(input_data_directory)

weather_writer = csv.writer(output_data_file)

for file in data_files:
    if not file.startswith("."):
        with open(os.path.join(input_data_directory, file)) as file_contents:
            file_json_data = json.load(file_contents)
            time_stamp = file_json_data['timestamp']
            for record in file_json_data['payload']:
                print(record)
                weather = " ".join([desc["description"] for desc in record['weather']["weather"]])
                city = record["city"]
                temp = record['weather']['main']["temp"]
                temp_min = record['weather']['main']["temp_min"]
                temp_max = record['weather']['main']["temp_max"]
                pressure = record['weather']['main']["pressure"]
                humidity = record['weather']['main']["humidity"]
                visibility = record['weather']["visibility"]
                wind_speed = record['weather']["wind"]["speed"]
                clouds_all = record['weather']["clouds"]["all"]
                sunrise = record['weather']["sys"]["sunrise"]
                sundown = record['weather']["sys"]["sunset"]
                weather_writer.writerow(
                    [
                        time_stamp,
                        city,
                        temp,
                        temp_min,
                        temp_max,
                        pressure,
                        humidity,
                        visibility,
                        wind_speed,
                        clouds_all,
                        weather,
                        sunrise,
                        sundown
                    ]
                    
                )

output_data_file.close()

import csv
import json
import os
import shutil
import datetime

input_data_directory = "data/select_data/weather.csv"
output_data = "data/preprocess/weather.csv"
try:
    os.remove(output_data)
except:
    pass
output_data_file = open(output_data, "w+")
weather_writer = csv.writer(output_data_file)

# '%Y-%m-%d_%H-%M-%S'

with open(input_data_directory) as csvfile:
    weather_reader = csv.reader(csvfile)
    for row in weather_reader:
        (
            time_stamp, city, temp, 
            temp_min, temp_max, pressure, 
            humidity, visibility, wind_speed, 
            clouds_all, weather, sunrise, 
            sundown
        ) = row
        
        date_time_weather = datetime.datetime.strptime(time_stamp, '%Y-%m-%d_%H-%M-%S')
        sunrise_date = datetime.datetime.fromtimestamp(int(row[-2]))
        sunset_date = datetime.datetime.fromtimestamp(int(row[-1]))
        sunrise_seconds = int((date_time_weather - sunrise_date).total_seconds())
        sunset_seconds = int((date_time_weather - sunset_date).total_seconds())

        
        weather_writer.writerow(
            [
                time_stamp, city, temp, 
                temp_min, temp_max, pressure, 
                humidity, visibility, wind_speed, 
                clouds_all, weather, sunrise_seconds, 
                sunset_seconds,
            ]
        )


output_data_file.close()

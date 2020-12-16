import csv
import json
import os
import shutil
import datetime

input_data_directory = "data/preprocess/weather.csv"
output_data = "data/transform_data/merged_data.csv"
try:
    os.remove(output_data)
except:
    pass
output_data_file = open(output_data, "w+")
weather_writer = csv.writer(output_data_file)

# '%Y-%m-%d_%H-%M-%S'

with open(input_data_directory) as weather_csvfile:
    weather_reader = list(csv.reader(weather_csvfile))
    for weather_row in weather_reader:
        (
            time_stamp_weather, city_weather, temp_weather, 
            temp_min_weather, temp_max_weather, pressure_weather, 
            humidity_weather, visibility_weather, wind_speed_weather, 
            clouds_all_weather, weather_weather, sunrise_weather, 
            sundown_weather,
        ) = weather_row
        with open("data/preprocess/bikes.csv") as bikes_csvfile:
            bikes_reader = csv.reader(bikes_csvfile)
            for bike_row in bikes_reader:
                (
                    time_bikes, city_bikes, address_bikes, banking_bikes, 
                    bonus_bikes, status_bikes, bike_stands_bikes, 
                    available_bike_stands_bikes, available_bikes_bikes,
                ) = bike_row
                date_weather = datetime.datetime.strptime(time_stamp_weather, '%Y-%m-%d_%H-%M-%S')
                date_bikes = datetime.datetime.strptime(time_bikes, '%Y-%m-%d_%H-%M-%S')
                date_weather_merge = datetime.datetime(
                    year=date_weather.year, 
                    month=date_weather.month, 
                    day=date_weather.day, 
                    hour=date_weather.hour
                )
                date_bikes_merge = datetime.datetime(
                    year=date_bikes.year, 
                    month=date_bikes.month, 
                    day=date_bikes.day, 
                    hour=date_bikes.hour
                )
                print(date_bikes_merge)
                print(date_weather_merge)
                print(city_weather)
                print(city_bikes)
                print(date_weather_merge == date_bikes_merge)

                if city_weather == city_bikes and date_weather_merge == date_bikes_merge:
                    weather_writer.writerow(
                        [
                            time_stamp_weather, city_weather, temp_weather, 
                            temp_min_weather, temp_max_weather, pressure_weather, 
                            humidity_weather, visibility_weather, wind_speed_weather, 
                            clouds_all_weather, weather_weather, sunrise_weather, 
                            sundown_weather, address_bikes, banking_bikes, 
                            bonus_bikes, status_bikes, bike_stands_bikes, 
                            available_bike_stands_bikes, available_bikes_bikes
                        ]
                    )
                if date_weather_merge < date_bikes_merge:
                    break


output_data_file.close()

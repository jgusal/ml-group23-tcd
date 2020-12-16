import csv
import json
import os
import shutil
import datetime

input_data_directory = "data/ml_scraper/json"
output_data = "data/select_data/bikes.csv"
try:
    os.remove(output_data)
except:
    pass
output_data_file = open(output_data, "w+")
data_files = os.listdir(input_data_directory)

bikes_writer = csv.writer(output_data_file)

for file in data_files:
    if not file.startswith(".") and os.path.isfile(os.path.join(input_data_directory, file)):
        with open(os.path.join(input_data_directory, file)) as file_contents:
            file_json_data = json.load(file_contents)
            time_stamp = file_json_data['timestamp']
            for record in file_json_data['payload']:
                city = record["contract_name"]
                address = record["address"]
                banking = record["banking"]
                bonus = record["bonus"]
                status = record["status"]
                bike_stands = record["bike_stands"]
                available_bike_stands = record["available_bike_stands"]
                available_bikes = record["available_bikes"]
                
                date_bikes = datetime.datetime.strptime(time_stamp, '%Y-%m-%d_%H-%M-%S')
                date_bikes_merge = datetime.datetime(
                    year=date_bikes.year, 
                    month=date_bikes.month, 
                    day=date_bikes.day, 
                    hour=date_bikes.hour
                )
                print(date_bikes_merge)
                print(date_bikes_merge >= datetime.datetime(
                    year=2020, 
                    month=10, 
                    day=28
                ))

                if date_bikes_merge > datetime.datetime(
                    year=2020, 
                    month=10, 
                    day=28
                ):
                    pass
                
                    bikes_writer.writerow(
                        [
                            time_stamp,
                            city,
                            address,
                            banking,
                            bonus,
                            status,
                            bike_stands,
                            available_bike_stands,
                            available_bikes,
                        ]
                    )

output_data_file.close()

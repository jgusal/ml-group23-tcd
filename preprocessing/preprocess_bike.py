import csv
import os

input_data_directory = "../data/selected_data/bikes.csv"
output_data = "../data/preprocessed_data/bikes.csv"
try:
    os.remove(output_data)
except:
    pass
output_data_file = open(output_data, "w+")
bikes_writer = csv.writer(output_data_file)

with open(input_data_directory) as csvfile:
    bike_reader = csv.reader(csvfile)
    for row in bike_reader:
        time_stamp, city, address, banking, bonus, status, bike_stands, available_bike_stands, available_bikes = row
        status = True if status == "OPEN" else False
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

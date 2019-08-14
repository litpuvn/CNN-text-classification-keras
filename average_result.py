import csv
import statistics



def create_dates():
    my_dates = []
    for i in range(22, 30):
        my_day = str(i)
        my_dates.append('10-' + my_day)

    # my_dates.append('09-01')
    # my_dates.append('09-02')

    return my_dates


def average_date(filter_day:str, dataset:str):
    global data


    with open('need_data/' + dataset + '_score.csv') as file_pointer:
        reader = csv.reader(file_pointer, delimiter=',')
        for idx, row in enumerate(reader):
            data_day = row[1]
            score = float(row[8])

            if filter_day == data_day.strip():
                data.append(score)


    mean_val = statistics.mean(data)
    print('average day:', filter_day, ':is:', mean_val)

    return mean_val
        # print('average city:', filter_city, ':is:', statistics.mean(city_data))

def average_city(filter_city:str, dataset:str,):

    global city_data

    with open('need_data/' + dataset + '_score.csv') as file_pointer:
        reader = csv.reader(file_pointer, delimiter=',')
        for idx, row in enumerate(reader):

            data_city = row[0]

            score = float(row[8])
            if filter_city is not None and filter_city == data_city.strip():
                city_data.append(score)

    print('average city:', filter_city, ':is:', statistics.mean(city_data))


dataset = 'sandy'
my_dates = create_dates()

daily_avg = []
for d in my_dates:
    data = []
    avg = average_date(d, dataset)
    daily_avg.append(avg)

print('daily average:', statistics.mean(daily_avg))
#
# filter_cities = ['houston', 'san-antonio', 'austin', 'dallas', 'corpus-christi']
# for city in filter_cities:
#     city_data = []
#
#     average_city(filter_city=city, dataset=dataset)
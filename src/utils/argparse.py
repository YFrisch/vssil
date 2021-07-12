import argparse
from datetime import datetime


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help='Absolute path to config file.', required=True)

    parser.add_argument('-d', '--data', type=str, help='Absoulte base path to dataset.', required=True)

    year, month, day, hour, minute = datetime.now().year, datetime.now().month, datetime.now().day, \
                                     datetime.now().hour, datetime.now().minute

    parser.add_argument('-i', '--id', type=str,
                        help='Individual id of the experiment.',
                        default=f"{year}_{month}_{day}_{hour}_{minute}")

    args = parser.parse_args()

    return args

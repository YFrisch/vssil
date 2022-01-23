import argparse
from datetime import datetime


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help='Absolute path to config file.', required=True)

    parser.add_argument('-d', '--data', type=str, help='Absolute base path to dataset.', required=True)

    year, month, day, hour, minute = datetime.now().year, datetime.now().month, datetime.now().day, \
                                     datetime.now().hour, datetime.now().minute

    parser.add_argument('-i', '--id', type=str,
                        help='Individual id of the experiment.',
                        default=f"{year}_{month}_{day}_{hour}_{minute}")

    parser.add_argument('-hp', '--hp', type=str,
                        help='Hyperparamter change (E.g. "training.epochs=10,data.num_workers=2"')

    parser.add_argument('-ch', '--checkpoint', type=str, help='Absoulte path to model checkpoints.')

    args = parser.parse_args()

    return args

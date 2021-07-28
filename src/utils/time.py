from datetime import datetime

START_TIME = None


def set_start_time():
    global START_TIME
    START_TIME = datetime.now()


def get_start_time():
    return START_TIME

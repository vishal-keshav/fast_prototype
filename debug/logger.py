"""
Logger module that:
1. Logs the training logs.
2. Sends log notifications to registered channels
"""

from statistics import mean
import ntfy
from ntfy.backends.slack import notify

class logger:
    def __init__(self, keys, send = False, token = None):
        self.keys = keys
        self.batch_log = {}
        self.epoch_log = []
        self.init_batch()
        self.notif = notification(token)
        self.send = send

    def init_batch(self):
        self.batch_log = {}
        for key in self.keys:
            self.batch_log.update({key: []})

    def batch_logger(self, data_dict, index):
        print("BATCH " + str(index))
        for key, value in data_dict.items():
            self.batch_log[key].append(value)
            print( key + " : " + str(value))

    def epoch_logger(self, index):
        print("Epoch " + str(index))
        self.epoch_log.append(self.batch_log)
        msg = "Epoch " + str(index) + " done with avg "
        for key, value in self.batch_log.items():
            mean_value = mean(list(map(float, value)))
            msg = msg + key + " : " + str(mean_value) + ".. "
        print(msg)
        self.init_batch()
        if self.send:
            self.notif.set_and_notify(msg)


class notification:
    def __init__(self, token):
        self.title = "fast_prototype"
        self.message = ""
        self.token = ""
        self.recipient = "#general"

    def set_message(self, msg):
        self.message = msg

    def send_notification(self):
        notify(self.title, self.message, token = self.token, recipient = self.recipient)

    def set_and_notify(self, msg):
        self.set_message(msg)
        self.send_notification()

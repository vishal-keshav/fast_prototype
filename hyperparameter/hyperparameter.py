# This file contains a class that has description and values of all hyper-parameters used for training.
import json
from pathlib import Path

class HyperParameters:
    def __init__(self, version, project_path):
        self.version = version
        self.project_path = project_path
        file = Path(project_path + "/hyperparameter/version" + str(version) + "/params.txt")
        if file.is_file():
            self.read_parameter()
        else:
            self.parameter_list = {}
            self.dump_parameter()

    def add_parameter(self, name, value):
        if name in self.parameter_list.keys():
            print("Already exists")
        else:
            self.parameter_list[name] = value

    def set_parameter(self, name, value):
        self.parameter_list[name] = value

    def get_value(self, name):
        return self.parameter_list[name]

    def dump_parameter(self):
        file_path = self.project_path + "/hyperparameter/version" + str(self.version) + "/params.txt"
        json.dump(self.parameter_list, open(file_path, "w"))

    def read_parameter(self):
        file_path = self.project_path + "/hyperparameter/version" + str(self.version) + "/params.txt"
        self.parameter_list = json.load(open(file_path))

    def read_parameter_space(self):
        file_path = self.project_path + "/hyperparameter/version" + str(self.version) + "/param_space.txt"
        self.parameter_space = json.load(open(file_path))

    def get_params(self):
        return self.parameter_list

    def get_param_space(self):
        file = Path(self.project_path + "/hyperparameter/version" + str(self.version) + "/param_space.txt")
        if file.is_file():
            self.read_parameter_space()
        else:
            self.parameter_space = {}
        return self.parameter_space

    def reset_params(self):
        self.parameter_list = {}
        self.dump_parameter()

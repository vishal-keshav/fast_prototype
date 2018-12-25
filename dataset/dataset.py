"""
An interface that interacts with the dataset specific scripts for:
1. Downloading the raw data
2. Pre-processing the raw data
3. View the pre-processed data

"""
import importlib
import keyboard

class DataSet:
    def __init__(self, dataset_name, absolute_path):
        self.dataset_name = dataset_name
        self.absolute_path = absolute_path
        dataset_module_path = "dataset." + self.dataset_name + ".dataset_script"
        dataset_module = importlib.import_module(dataset_module_path)
        self.dataset_obj = dataset_module.get_obj(absolute_path)
        data_provider_module_path = "dataset." + self.dataset_name + ".data_provider"
        data_provider_module = importlib.import_module(data_provider_module_path)
        self.data_provider = data_provider_module.get_obj(absolute_path)

    def get_data(self):
        self.dataset_obj.get_data()

    def preprocess_data(self):
        self.dataset_obj.preprocess_data()

    def view(self):
        print("Press forward arrowkey for next, q to end")
        dp = self.data_provider
        dp.set_batch(1)
        out = dp.next()
        dp.view(out)
        # TODO: Key stroke based loop
        """while True:
            try:
                if keyboard.is_pressed('q'):
                    break
                else:
                    out = dp.next()
                    dp.view(out)
            except:
                break"""

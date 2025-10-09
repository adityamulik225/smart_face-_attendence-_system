# utils.py मध्ये ठेवा
import json
import os

class Conf:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file '{path}' not found")
        with open(path, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, key):
        return self.data[key]

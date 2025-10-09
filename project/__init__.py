import json

class Conf:
    def __init__(self, confPath):
        # JSON फाईल लोड करतो
        with open(confPath) as f:
            self._conf = json.load(f)

    def __getitem__(self, k):
        return self._conf.get(k, None)

class CollectionDescription(object):
    def __init__(self, keys, values):
        for k, v in zip(keys, values):
            self.__dict__[k] = v

    def __str__(self):
        return str(self.__dict__)

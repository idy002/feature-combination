from .Couple import Couple


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    if data_name == 'couple':
        return Couple(initialized=initialized)
    else:
        raise ValueError

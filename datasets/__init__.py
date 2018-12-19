from .Couple import Couple
from .ml1m import ml1m
from .MovieLens import MovieLens


def as_dataset(data_name, initialized=True):
    data_name = data_name.lower()
    if data_name == 'couple':
        return Couple(initialized=initialized)
    elif data_name == 'ml1m':
        return ml1m(initialized=initialized)
    elif data_name == 'movielens':
        return MovieLens(initialized=initialized)
    else:
        raise ValueError

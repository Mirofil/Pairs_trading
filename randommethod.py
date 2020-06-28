from distancemethod import distance
import random
import pandas as pd
from typing import *

def random_pairs(df: pd.DataFrame, num: int = 20):
    distance_pairs = distance(df, num=1000000000)
    random_pairs = random.sample(distance_pairs[2], num)
    return random_pairs
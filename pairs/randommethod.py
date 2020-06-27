from pairs.distancemethod import distance
import random

def random_pairs(df: pd.DataFrame, num: int = 20, method="modern", show_progress_bar=True):
    distance_pairs = distance(df, num=1000000000, method='modern', show_progress_bar=show_progress_bar)
    random_pairs = random.sample(distance_pairs['viable_pairs'], num)
    return random_pairs
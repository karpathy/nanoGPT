import numpy as np
import pandas as pd


class ExportData:
    def __init__(self) -> None:
        self.array = []

    def add(self, iteration, loss, time, mfu):
        self.array.append([
            iteration, loss, time, mfu])

    def save(self, out_dir):
        df = pd.DataFrame(data=self.array, columns=[
                          'iteration', 'loss', 'time', 'mfu'])
        df.set_index('iteration', inplace=True)
        df.to_csv(f'{out_dir}/data.csv')



import pandas as pd


class ExportData:
    def __init__(self) -> None:
        self.array = []

    def add(self, iteration, train_loss, val_loss, lr, mfu):
        self.array.append([
            iteration, train_loss, val_loss, lr, mfu])

    def save(self, out_dir):
        df = pd.DataFrame(data=self.array, columns=[
                          'iteration', 'train_loss', 'val_loss', 'lr', 'mfu'])
        df.set_index('iteration', inplace=True)
        df.to_csv(f'{out_dir}/data.csv')



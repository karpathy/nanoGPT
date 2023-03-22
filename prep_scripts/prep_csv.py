import pandas as pd

column = 'Introduction: Background'
input_filename = 'cia_world.csv'
output_filename = '../datasets/dataset_cia_world.txt'


df = pd.read_csv(input_filename)
# print(df.columns)
df[column].to_csv(output_filename, index=False, header=False)

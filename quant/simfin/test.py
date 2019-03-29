import pandas as pd

table = pd.read_pickle('data/table.zip')

flws = table.query('Ticker == "FLWS"')



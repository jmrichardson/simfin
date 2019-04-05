
from importlib import reload
import simfin
out = reload(simfin)
from simfin import *

# df = pd.read_pickle('tmp/extract.zip')
# df = df.query('Ticker == "FLWS"')

sf = SimFin().flatten()
sf.data_df.to_csv('look.csv')
df = sf.data_df
df = df.query('Ticker == "A"')
lag = -4
field = 'Flat_SPQA'

new = sf.target()
df = new.data_df
new.data_df.to_csv('look.csv')



# sf = SimFin().flatten().query(['FLWS','TSLA']).target().process().save('rf')
df = SimFin().flatten().query(['FLWS','TSLA']).data_df
sf = SimFin().flatten().target().process().save('rf')


# df = SimFin().flatten().query(['FLWS','TSLA']).target().data_df
df = SimFin().flatten().query(['FLWS']).target().data_df

df = SimFin().flatten().features().target().process().save('rf2')

df = SimFin().flatten().query(['FLWS','TSLA']).target().process().data_df


df = SimFin().load('rf').data_df
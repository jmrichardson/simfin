
from simfin import *

sf = SimFin().flatten().target().process().save('rf')
# sf = SimFin().flatten().query(['FLWS','TSLA']).target().process().save('rf')


# df = SimFin().flatten().query(['FLWS','TSLA']).target().data_df
df = SimFin().flatten().query(['FLWS','TSLA']).features().target().data_df
df = SimFin().flatten().features().target().process().save('rf2')

df = SimFin().flatten().query(['FLWS','TSLA']).target().process().data_df


df = SimFin().load('rf').data_df
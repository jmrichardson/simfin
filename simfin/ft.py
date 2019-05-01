import pandas as pd
import featuretools as ft
from config import *


key_features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'EBITDA', 'Net Profit',
            'Cash & Cash Equivalents', 'Receivables', 'Current Assets',  'Total Assets', 'Short term debt', 'Accounts Payable',
            'Current Liabilities', 'Long Term Debt', 'Total Liabilities', 'Share Capital', 'Total Equity',
            'Free Cash Flow', 'Gross Margin', 'Operating Margin', 'Net Profit Margin', 'Return on Equity',
            'Return on Assets', 'Current Ratio', 'Liabilities to Equity Ratio', 'Debt to Assets Ratio',
            'EV / EBITDA', 'EV / Sales', 'Book to Market', 'Operating Income / EV', 'Enterprise Value',
            'Flat_Basic Earnings Per Share', 'Flat_Common Earnings Per Share', 'Flat_Diluted Earnings Per Share',
            'Flat_Basic PE', 'Flat_Common PE', 'Flat_Diluted PE']

key_features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'EBITDA', 'Net Profit',
            'Cash & Cash Equivalents', 'Receivables', 'Current Assets',  'Total Assets', 'Short term debt', 'Accounts Payable',
            'Current Liabilities', 'Long Term Debt', 'Total Liabilities', 'Share Capital', 'Total Equity',]

# key_features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'EBITDA', 'Net Profit']
# key_features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'Net Profit']
# key_features = ['Revenues', 'COGS', 'SG&A']

data = X.loc[:, key_features]
d = ft.primitives.list_primitives()

# Make an entityset and add the entity
es = ft.EntitySet(id='simfin')
es.entity_from_dataframe(entity_id='simfin', dataframe=data,
                         make_index=True, index='index')

# Run deep feature synthesis with transformation primitives
fm, fd = ft.dfs(entityset=es, target_entity='simfin',
                trans_primitives=['add_numeric', 'divide_numeric'],
                # features_only=True,
                # n_jobs=2,
                # max_features=2,
                verbose=True)

# fm
fd
#!/usr/bin/env python

import pandas as pd
import os

abspath = os.path.dirname(__file__)

# open fer and ferplus csv files
fer = pd.read_csv(abspath + '/../data/fer2013.csv')
fernew = pd.read_csv(abspath + '/../data/fer2013new.csv')

# copy pixels to new csv
fernew['Image name'] = fer['pixels']

# rename column
new_columns = fernew.columns.values
new_columns[1] = 'pixels'
fernew.columns = new_columns

# write csv
fernew.to_csv(abspath + '/../data/ferplus.csv', index=False)

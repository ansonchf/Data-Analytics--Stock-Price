import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

filename = 'E:\\Nifty\\Nifty.xlsx'
f = open(filename, 'r')
#f.readlines()[:5] # Shows the first five rows of the raw data.
# The filename passed to the pd.read_csv() function.
data = pd.read_excel(filename,
                   index_col=0, # Specifies that the first column shall be handled as an index.
                   parse_dates=True) # Specifies that the index values are of type datetime.
data.info() # The resulting DataFrame object.

data.head()

data.tail()


data.plot(figsize=(10, 12), subplots=True);

#%%


# Summary Statistics
# info() gives some metainformation about the DataFrame object.
data.info()
# describe() provides useful standard statistics per column.
data.describe().round(2)

# Changes over Time
# diff() provides the absolute changes between two index values.
data.diff().head()
# Aggregation operations can be applied in addition.
data.diff().mean()

# pct_change() calculates the percentage change
# between two index values.
data.pct_change().round(3).head()
# The mean values of the results are visualized as a bar plot
data.pct_change().mean().plot(kind='bar', figsize=(10, 6));

# Calculates the log returns in vectorized fashion.
rets = np.log(data / data.shift(1))
# A subset of the results.
rets.head().round(3)
# Plots the cumulative log returns over time
rets.cumsum().apply(np.exp).plot(figsize=(10, 6));


#%% Correlation Analysis

# Import library and data
import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

# Reads the EOD data (originally from the Thomson Reuters Eikon Data API) from a CSV file.
data = pd.read_excel(filename,
                   index_col=0, # Specifies that the first column shall be handled as an index.
                   parse_dates=True) # Specifies that the index values are of type datetime.

# .loc[:DATE] selects the data until the given value DATE.
data.loc[:'2019-12-31'].plot(secondary_y='Return', figsize=(10, 6));

rets = np.log(data / data.shift(1))
rets.head()
rets.dropna(inplace=True)
rets.plot(subplots=True, figsize=(10, 6));
pd.plotting.scatter_matrix(rets, # The data set to be plotted.
                           alpha=0.5, # The alpha parameter for the opacity of the dots.
                           diagonal='kde', # What to place on the diagonal; here: a histogram of the column data.
                           hist_kwds={'bins': 35}, # Keywords to be passed to the histogram plotting function.
                           figsize=(10, 6));
                           
                           
#%%                          
# This implements a linear OLS regression.

rets.corr() # The correlation matrix for the whole DataFrame.                           
                           
reg = np.polyfit(rets['Return'], rets['PE'], deg=1)
# This plots the log returns as a scatter plot to which the linear regression line is added.
ax = rets.plot(kind='scatter', x='Return', y='PE', figsize=(10, 6))
ax.plot(rets['Return'], np.polyval(reg, rets['Return']), 'r', lw=2);

reg = np.polyfit(rets['Return'], rets['PB'], deg=1)
# This plots the log returns as a scatter plot to which the linear regression line is added.
ax = rets.plot(kind='scatter', x='Return', y='PB', figsize=(10, 6))
ax.plot(rets['Return'], np.polyval(reg, rets['Return']), 'r', lw=2);

reg = np.polyfit(rets['Return'], rets['DIV'], deg=1)
# This plots the log returns as a scatter plot to which the linear regression line is added.
ax = rets.plot(kind='scatter', x='Return', y='DIV', figsize=(10, 6))
ax.plot(rets['Return'], np.polyval(reg, rets['Return']), 'r', lw=2);

reg = np.polyfit(rets['Return'], rets['EPS'], deg=1)
# This plots the log returns as a scatter plot to which the linear regression line is added.
ax = rets.plot(kind='scatter', x='Return', y='EPS', figsize=(10, 6))
ax.plot(rets['Return'], np.polyval(reg, rets['Return']), 'r', lw=2);

#%%         
# This plots the rolling correlation over time …
ax = rets['Return'].rolling(window=252).corr(rets['PE']).plot(figsize=(10, 6))
# … and adds the static value to the plot as horizontal line.
ax.axhline(rets.corr().iloc[0, 1], c='r');

ax = rets['Return'].rolling(window=252).corr(rets['PB']).plot(figsize=(10, 6))
# … and adds the static value to the plot as horizontal line.
ax.axhline(rets.corr().iloc[0, 1], c='r');

ax = rets['Return'].rolling(window=252).corr(rets['DIV']).plot(figsize=(10, 6))
# … and adds the static value to the plot as horizontal line.
ax.axhline(rets.corr().iloc[0, 1], c='r');

ax = rets['Return'].rolling(window=252).corr(rets['EPS']).plot(figsize=(10, 6))
# … and adds the static value to the plot as horizontal line.
ax.axhline(rets.corr().iloc[0, 1], c='r');

#%%

import pandas as pd
import statsmodels.api as sm
Nifty=pd.read_excel('E:\\Nifty\\Nifty.xlsx')
Nifty.head()

Nifty.set_index('Time',inplace=True)
Nifty.head()

XVar=Nifty.drop('Return', axis=1)
YVar=Nifty[['Return']]
YVar.head()
XVar.head()

linearModel1=sm.OLS(YVar, XVar).fit()
print(linearModel1.summary())

XVar2=sm.add_constant(XVar)
linearModel2=sm.OLS(YVar, XVar2).fit()
print(linearModel2.summary())


#%%

import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
x = np.asarray(XVar)
y = np.asarray(YVar)

# =============================================================================

xt = x.transpose()
x0 = xt[0]
x1 = xt[1]
x2 = xt[2]
x3 = xt[3]

#xt = x.transpose()(0)
yt = y.transpose()[0]

reg = np.polyfit(x0, yt, 1)
reg
help(np.polyfit)

plt.figure(figsize=(10, 6))
plt.scatter(x0, yt, c=yt, marker='v', cmap='coolwarm')
plt.plot(x0, reg[1] + reg[0] * x0, lw=2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

reg = np.polyfit(x1, yt, 1)
reg
help(np.polyfit)

plt.figure(figsize=(10, 6))
plt.scatter(x1, yt, c=yt, marker='v', cmap='coolwarm')
plt.plot(x1, reg[1] + reg[0] * x1, lw=2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')


reg = np.polyfit(x2, yt, 1)
reg
help(np.polyfit)

plt.figure(figsize=(10, 6))
plt.scatter(x2, yt, c=yt, marker='v', cmap='coolwarm')
plt.plot(x2, reg[1] + reg[0] * x2, lw=2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')


reg = np.polyfit(x3, yt, 1)
reg
help(np.polyfit)

plt.figure(figsize=(10, 6))
plt.scatter(x3, yt, c=yt, marker='v', cmap='coolwarm')
plt.plot(x3, reg[1] + reg[0] * x3, lw=2.0)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

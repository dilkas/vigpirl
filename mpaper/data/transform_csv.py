import pandas
import numpy

filename = 'median_Gamma'

def normalise(data):
    transformed = numpy.log(data)
    max_value = transformed.max().max()
    min_value = transformed.min().min()
    return transformed.transform(lambda x: 1000 * (x - min_value) / (max_value - min_value))

def normalise2(data):
    '''Scale positive and negative values differently, so that both -1 and 1 are reached.'''
    max_value = data.max().max()
    min_value = data.min().min()
    return data.applymap(lambda x: 500+ 500 * x / max_value if x > 0 else 500 + 500 * x / (-min_value))

df = pandas.read_csv(filename + '.csv', header=None)
indices = df.iloc[:,:2]
diagonal = df.iloc[:,[2,6,10]]
nondiagonal = df.iloc[:,[3,4,7]]
#policy = df.iloc[:,[11,12,16]]

#together = pandas.concat([indices, normalise(diagonal), normalise(nondiagonal), policy], axis=1)
together = pandas.concat([indices, normalise2(diagonal), normalise2(nondiagonal)], axis=1)
together.to_csv(filename + '_transformed.csv', header=None, index=False)

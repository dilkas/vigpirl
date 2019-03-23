import pandas

def normalise(data):
    max_value = data.transform(lambda x: abs(x)).max().max()
    return data.transform(lambda x: 500 * x / max_value + 500)

df = pandas.read_csv('covariance_and_policy.csv', header=None)
indices = df.iloc[:,:2]
diagonal = df.iloc[:,[2,6,10]]
nondiagonal = df.iloc[:,[3,4,7]]
policy = df.iloc[:,[11,12,16]]

together = pandas.concat([indices, normalise(diagonal), normalise(nondiagonal), policy], axis=1)
together.to_csv('covariance_and_policy_transformed.csv', header=None, index=False)

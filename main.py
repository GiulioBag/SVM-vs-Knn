from sklearn import datasets
from sklearn.utils import shuffle

from skrebate import ReliefF, SURF
from utility import *

'''Points 1-3'''

wine = datasets.load_wine()
x = wine.data[:, :2]
y = wine.target

x, y = shuffle(x, y, random_state=483)

scaler = StandardScaler()
scaler.fit(x[:int(len(x) * 0.7)])

sets = makeSets(x, y)

feature_name = [wine.feature_names[0], wine.feature_names[1]]

'''Points 4-18'''

homework(sets[0], sets[1], sets[2], sets[3], sets[4], sets[5], sets[6], sets[7], feature_name)

'''Point 20'''

features = wine.data
train_label = wine.target
features, train_label = shuffle(features, train_label, random_state=483)

scaler = MinMaxScaler()
scaler.fit(features[:int(len(features) * 0.7)])

train_features = scaler.transform(features[: int(len(features) * 0.7)])
train_label = train_label[: int(len(y) * 0.7)]

fs = ReliefF(n_features_to_select=2, n_neighbors=178)
fs = fs.fit(train_features, train_label)
top = fs.top_features_

for index in range(13):
    print(index, ") ", wine.feature_names[top[index]], " ---- ", fs.feature_importances_[top[index]])


x = np.concatenate((wine.data[:, top[0]:top[0] + 1], wine.data[:, top[1]:top[1] + 1]), axis=1)
y = wine.target
x, y = shuffle(x, y, random_state=483)

sets = makeSets(x, y)

feature_name = [wine.feature_names[top[0]], wine.feature_names[top[1]]]
homework(sets[0], sets[1], sets[2], sets[3], sets[4], sets[5], sets[6], sets[7], feature_name)


x = np.concatenate((wine.data[:, top[12]:top[12] + 1], wine.data[:, top[11]:top[11] + 1]), axis=1)
y = wine.target
x, y = shuffle(x, y, random_state=483)

sets = makeSets(x, y)

feature_name = [wine.feature_names[top[12]], wine.feature_names[top[11]]]
homework(sets[0], sets[1], sets[2], sets[3], sets[4], sets[5], sets[6], sets[7], feature_name)

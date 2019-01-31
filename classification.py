from sklearn.svm import SVC
import pandas as pd

# read the data from the csv file
df = pd.read_csv("..\\features.csv", header=None)
X = df.values[:, 0:-1]
y = df.values[:, -1]

print("start training")
model = SVC(C=10000.0, gamma=1000, random_state=42, kernel="rbf")
model.fit(X, y)
print("Done")
print(model.score(X, y))

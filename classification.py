from sklearn.svm import SVC
from features_extraction import get_lbp


features, labels = get_lbp()
print("start training")
model = SVC(C=100.0, random_state=42, max_iter=10000, kernel="rbf")
model.fit(features, labels)
print("Done")
print(model.score(features, labels))
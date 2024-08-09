from tsai.inference import load_learner
from tsai.basics import *

X, y, splits = get_classification_data('ECG200', split_data=False)
tfms = [None, TSClassification()]
print(tfms)
batch_tfms = TSStandardize()
clf = TSClassifier(X, y, splits=splits, path='models', arch="InceptionTimePlus",
                   tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, device='cuda')
print(clf.model)
clf.fit_one_cycle(100, 3e-4)
clf.plot_metrics()
clf.export("clf.pkl")


clf = load_learner("models/clf.pkl", cpu=False)
probas, target, preds = clf.get_X_preds(X[splits[1]], y[splits[1]])
print(np.array(preds, dtype=int))
print(f'{probas=}, {target=}, {preds=}')
# print(acc)

# from tsai.models.InceptionTime import InceptionTime

# dsid = 'NATOPS'
# X, y, splits = get_UCR_data(dsid, return_split=False)
# tfms = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[
#                                64, 128], batch_tfms=[TSStandardize()], num_workers=0)
# model = InceptionTime(dls.vars, dls.c)
# learn = Learner(dls, model, metrics=accuracy)
# learn.fit_one_cycle(25, lr_max=1e-3)
# learn.plot_metrics()

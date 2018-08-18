#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import itertools
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def corr_df(features, corr_val):
    '''
....Filters out features based on pearson correlation coefficent
....returns a list of indexes which can be removed
....'''

    corr_matrix = np.corrcoef(features, rowvar=False)
    print ('Correlation Matrix Created')
    drop_cols = set()
    (row, columns) = corr_matrix.shape
    for i in range(row):
        for j in range(i + 1, columns):
            if corr_matrix[i][j] > corr_val or corr_matrix[i][j] \
                < corr_val * -1:

                # print(corr_matrix[i][j], i , j)

                drop_cols.add(i)
                drop_cols.add(j)
    return drop_cols


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues,
    ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print ('Normalized confusion matrix')
    else:
        print ('Confusion matrix, without normalization')

    print (cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ('.2f' if normalize else 'd')
    thresh = cm.max() / 2.
    for (i, j) in itertools.product(range(cm.shape[0]),
                                    range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center', color=('white' if cm[i,
                 j] > thresh else 'black'))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# File/data processing
# load csv files into handlers

data_folder = 'TCGA-PANCAN-HiSeq-801x20531'

# import labels

file_handle = open(data_folder + '/labels.csv')
cls_can = dict()
labels = dict()
flag = True
count = 0
print ('Reading File...')
for line in file_handle:
    if flag:
        flag = False
        continue
    line = line.strip().split(',')
    if line[1] not in cls_can:
        cls_can[line[1]] = count
        labels[line[0]] = cls_can[line[1]]
        count += 1
    else:
        labels[line[0]] = cls_can[line[1]]
file_handle.close()
handle=open("class_id.csv", "w")
for key in cls_can:
	handle.write(key+','+str(cls_can[key])+'\n')
handle.close()
# import data

file_handle = open(data_folder + '/data.csv')
flag = True
features = list()
targets = list()
gene_id = list()
gene_list = list()

for line in file_handle:
    if flag:
        flag = False
        gene_list = line.strip().split(',')[1:]
        continue
    line = line.strip().split(',')
    for i in line[1:]:
        if i.strip() == '':
            print (line[0])
            continue
    targets.append(labels[line[0]])
    features.append(list(map(float, line[1:])))
file_handle.close()

# convert list into numpy array for model_fitting

targets = np.array(targets)
features = np.array(features)
gene_id = np.arange(0, len(features[0]))

#################################
###### DATA PREPROCESSING #######
#################################

print ('Pre-processing data......')

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.4, stratify=targets, random_state=42)

# normalise the features

scaler = preprocessing.StandardScaler().fit(x_train)
scaled_features = scaler.fit_transform(x_train)

# remove features with 0 variance

select = VarianceThreshold().fit(scaled_features)

selected_features = select.transform(scaled_features)

x_test = select.transform(x_test)

indexes = select.get_support()
gene_id = gene_id[indexes]

print ('Number of Samples', selected_features.shape[0])
print ('Number of Features', selected_features.shape[1])

print ('Filtering features based on correlation threshold')

# filter features using correlation threshold

handle = open(data_folder + '/index_70.csv')
filter_index = handle.readline().strip().split(',')
filter_index = [int(index) for index in filter_index]

# print(filter_index)
# filter_index=corr_df(selected_features, 0.70)
# filter_index1=[str(index) for  index in filter_index]
# handle.write(','.join(filter_index1))
# handle.close()
# print(filter_index)

selected_features = np.delete(selected_features, list(filter_index), 1)

x_test = np.delete(x_test, list(filter_index), 1)

gene_id = np.delete(gene_id, list(filter_index))

print ('Filtered Features based on high correlation coeffient of 0.70',
       selected_features.shape[1])

##################################################################################################
##### Filter features using RandomForestClassifier as prediction engine done multiple times ######
##################################################################################################

print ('Filtering Features based on Random Forest...')
select = None
counter = 0
min_v = 1.0
esti = RandomForestClassifier(n_estimators=100, n_jobs=3,
                              criterion='entropy',
                              class_weight='balanced', random_state=42)

while min_v > 0.98 and selected_features.shape[1] > 15:
    select = SelectFromModel(esti)
    select.fit(selected_features, y_train)
    selected_features = select.transform(selected_features)
    x_test = select.transform(x_test)
    indexes = select.get_support()
    gene_id = gene_id[indexes]
    counter = counter + 1
    (unique, counts) = np.unique(targets, return_counts=True)
    y_counts = dict(zip(unique, counts))
    scores = cross_val_score(
        estimator=esti,
        X=selected_features,
        y=y_train,
        scoring='accuracy',
        cv=min(y_counts, key=y_counts.get),
        n_jobs=3,
        )
    min_v = scores.min()
    print ('Features left after %d iterations %d with minimum score of %0.2f (+/- %0.2f)' \
        % (counter, selected_features.shape[1], min_v, scores.std() * 2))

selected_genes = list()
for i in gene_id:
    selected_genes.append(gene_list[i])

select.fit(selected_features, y_train)
imp_feature = select.estimator_.feature_importances_
fig = plt.figure()
plt.tight_layout()
plt.bar(selected_genes, imp_feature)
plt.xticks(rotation=70)
plt.ylabel('Feature Importance')
plt.xlabel('Gene ID')
plt.title('Importance of Selected Genes')
plt.savefig('feature_importance.png', bbox_inches='tight',
            transparent=True)
plt.close(fig)

print ('Selected Genes:')
selected_genes = list()
for i in gene_id:
    selected_genes.append(gene_list[i])
handle=open("selected_genes.csv", "w")
handle.write(','.join(selected_genes))
handle.close()
print ('\t'.join(selected_genes))
print ('Number of Selected Genes: %d' % len(selected_genes))

# Fitting a SupportVectorClassifier for a final model

clf = SVC(class_weight='balanced')

# split data into test and training data 20% test and 80%train

(unique, counts) = np.unique(y_train, return_counts=True)
y_counts = dict(zip(unique, counts))
print(y_counts)
scores = cross_val_score(
    estimator=clf,
    X=selected_features,
    y=y_train,
    scoring='accuracy',
    cv=min(y_counts, key=y_counts.get),
    n_jobs=3,
    )
print ('Training Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
(unique, counts) = np.unique(y_test, return_counts=True)
y_counts = dict(zip(unique, counts))
clf.fit(selected_features, y_train)
x_test = scaler.fit_transform(x_test)
scores = cross_val_score(
    estimator=clf,
    X=x_test,
    y=y_test,
    scoring='accuracy',
    cv=min(y_counts, key=y_counts.get),
    n_jobs=3,
    )
print ('Testing Accuracy Score %0.2f (+/- %0.2f)' % (scores.mean(),
        scores.std() * 2))
target_name = ['NA'] * len(cls_can)
for i in cls_can:
    target_name[cls_can[i]] = i
y_pred=clf.predict(x_test)
print (metrics.classification_report(y_test, y_pred,
                                    target_names=target_name))

# Compute confusion matrix

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix

plt.figure()
plt.tight_layout()
plot_confusion_matrix(cnf_matrix, classes=target_name, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('normalised_confusion_matrix.png', bbox_inches='tight',
            transparent=True)

from sklearn.externals import joblib
joblib.dump(clf, 'cancer_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')

			

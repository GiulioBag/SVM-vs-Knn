import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import neighbors, svm
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def multiDecisionB(train_x, validation_x, validation_y, title, clf, ax, feature_name):
    x1_min, x1_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    x2_min, x2_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    prediction = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    prediction = prediction.reshape(xx.shape)

    ax.contourf(xx, yy, prediction, cmap=cmap_light, alpha=0.9)
    ax.scatter(validation_x[:, 0], validation_x[:, 1], c=validation_y, cmap=cmap_bold, edgecolor='k',
               s=20)
    ax.set_ylabel(feature_name[0])
    ax.set_xlabel(feature_name[1])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)


def makeDecisionBoundary(features, validation_x, validation_y, clf, title, feature_name):
    x1_min, x1_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    x2_min, x2_max = features[:, 1].min() - 1, features[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    prediction = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    prediction = prediction.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, prediction, cmap=cmap_light)

    plt.scatter(validation_x[:, 0], validation_x[:, 1], c=validation_y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel(feature_name[0])
    plt.ylabel(feature_name[1])
    plt.show()


def makeAccuracyGraohic(values_x, values_y, title, xlabel, ylabel, scale="linear"):
    plt.plot(values_x, values_y, marker="o")
    plt.xscale(scale)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def applyKNN(train_x, train_y, validation_x, validation_y, test_x, test_y, feature_name):
    nums_Neighbors = [1, 3, 5, 7]
    score_validation = []
    best_model = {'num_Neighbors': 0, 'score': 0}

    for num_Neighbors in nums_Neighbors:

        clf = neighbors.KNeighborsClassifier(num_Neighbors)
        clf.fit(train_x, train_y)

        title = "KNN (k = " + str(num_Neighbors) + ")"
        makeDecisionBoundary(train_x, validation_x, validation_y, clf, title, feature_name)

        score = clf.score(validation_x, validation_y)
        score *= 100
        score = round(score, 4)
        score_validation.append(score)

        if score > best_model['score']:
            best_model['num_Neighbors'] = num_Neighbors
            best_model['score'] = score
            best_model['clf'] = clf

        print("Score on the validation set is: ", score, "%. The model was trained with: k = ",
              num_Neighbors)

    makeAccuracyGraohic(nums_Neighbors, score_validation, "Accuracy on Validation Set", "Number of Neighbors",
                        "Accuracy")

    score_test = round(best_model['clf'].score(test_x, test_y) * 100, 4)
    print("Score on the test set is: ", score_test, "%. The model was trained with k = ",
          best_model['num_Neighbors'], "\n \n")


def applySVM_C_(train_x, train_y, validation_x, validation_y, test_x, test_y, kernel, feature_name):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    score_validation = []
    best_model = {'c': 0, 'score': 0, 'clf': 0}

    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(12.5, 14.5)
    index = 0

    for c in Cs:

        clf = svm.SVC(kernel=kernel, C=c)
        clf.fit(train_x, train_y)

        if c < 1:
            title = "Kernel = " + kernel + ", C = " + str(round(c, 3))
        else:
            title = "Kernel = " + kernel + ", C = " + str(int(c))

        multiDecisionB(train_x, validation_x, validation_y, title, clf, ax[int(index / 2)][index % 2], feature_name)
        index += 1

        score = clf.score(validation_x, validation_y)
        score *= 100
        score = round(score, 4)
        score_validation.append(score)

        if score > best_model['score']:
            best_model['c'] = c
            best_model['score'] = score
            best_model['clf'] = clf

        print("Score on the validation set is: " + str(
            score) + " The model was trained with: kernel = " + kernel + " and C = " + str(c))

    makeAccuracyGraohic(Cs, score_validation, "Accuracy on Validation Set", "C", "Accuracy", "log")

    param = "Kernel = " + kernel + ", C = " + str(best_model['c'])

    score_test = round(best_model['clf'].score(test_x, test_y) * 100, 4)
    print("Score on the test set is: " + str(score_test) + "%. The model was trained with: " + param + "\n \n")


def applyCrossFold(cross_train_x, cross_train_y, test_x, test_y, Cs, gamma):
    best_model = {'c': 0, 'gamma': 0, 'score': 0}

    for g in gamma:
        for c in Cs:

            clf = svm.SVC(kernel="rbf", C=c, gamma=g)
            accuracy = cross_val_score(clf, cross_train_x, cross_train_y, scoring='accuracy', cv=5)
            score = sum(accuracy) * 20

            print("Score on the validation set with Cross-Fold Validation is: " + str(
                score) + ". The model with kernel = RBF, C= " + str(c) + ", gamma= " + str(g))

            if score > best_model['score']:
                best_model['c'] = c
                best_model['gamma'] = g
                best_model['score'] = score

    param = "Kernel = RBF, C = " + str(best_model['c']) + ", Gamma = " + str(best_model['gamma'])

    clf = svm.SVC(kernel="rbf", C=best_model['c'], gamma=best_model['gamma'])
    clf.fit(cross_train_x, cross_train_y)
    score_test = clf.score(test_x, test_y)

    print("Score on the test set using 5-Fold Cross Validation is: " + str(
        round(score_test, 4) * 100) + "%. The model was trained with " + param + "\n \n")


def applySVM_C_g_(train_x, train_y, validation_x, validation_y, test_x, test_y, Cs, gamma, feature_name):
    score_validation = []
    best_model = {'c': 0, 'gamma': 0, 'score': 0, 'clf': 0}

    for c in Cs:

        fig, ax = plt.subplots(4, 2)
        fig.set_size_inches(12.5, 14.5)
        index = 0
        for g in gamma:

            clf = svm.SVC(kernel="rbf", C=c, gamma=g)
            clf.fit(train_x, train_y)

            if c < 1:
                title = "Kernel = RBF, C = " + str(round(c, 3))
            else:
                title = "Kernel = RBF, C = " + str(int(c))

            if g < 1:
                title += " Gamma = " + str(round(g, 3))
            else:
                title += " Gamma = " + str(int(g))

            multiDecisionB(train_x, validation_x, validation_y, title, clf, ax[int(index / 2)][index % 2], feature_name)
            index += 1

            score = clf.score(validation_x, validation_y)
            score *= 100
            score = round(score, 4)
            score_validation.append(score)

            if score > best_model['score']:
                best_model['c'] = c
                best_model['gamma'] = g
                best_model['score'] = score
                best_model['clf'] = clf

            print("Score on the validation set is: " + str(
                score) + ". The model was trained with: kernel = RBF, C = " + str(c) + ", gamma = " + str(g))

        plt.show()

    param = "Kernel = rbf, C = " + str(best_model['c']) + ", Gamma = " + str(best_model['gamma'])

    score_test = round(best_model['clf'].score(test_x, test_y), 4) * 100
    print("Score on the test set is: " + str(score_test) + "%. The model was trained with: " + param + "\n \n")


def homework(train_x, validation_x, test_x, cross_train_x, train_y, validation_y, test_y, cross_train_y, feature_name):
    '''Points 4-7'''

    applyKNN(train_x, train_y, validation_x, validation_y, test_x, test_y, feature_name)

    '''Points 8-11'''

    kernel = "linear"
    applySVM_C_(train_x, train_y, validation_x, validation_y, test_x, test_y, kernel, feature_name)

    '''Points 12-13'''

    kernel = "rbf"
    applySVM_C_(train_x, train_y, validation_x, validation_y, test_x, test_y, kernel, feature_name)

    '''Points 15'''

    gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]
    Cs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000]

    applySVM_C_g_(train_x, train_y, validation_x, validation_y, test_x, test_y, Cs, gamma, feature_name)

    '''Points 16-18'''
    applyCrossFold(cross_train_x, cross_train_y, test_x, test_y, Cs, gamma)


def makeSets(x, y):
    scaler = StandardScaler()
    scaler.fit(x[:int(len(x) * 0.7)])

    train_x = scaler.transform(x[:int(len(x) / 2)])
    validation_x = scaler.transform(x[int(len(x) / 2): int(len(x) * 0.7)])
    test_x = scaler.transform(x[int(len(x) * 0.7):])
    cross_train_x = scaler.transform(x[: int(len(x) * 0.7)])

    train_y = y[:int(len(x) / 2)]
    validation_y = y[int(len(x) / 2): int(len(x) * 0.7)]
    test_y = y[int(len(x) * 0.7):]
    cross_train_y = y[: int(len(x) * 0.7)]

    return [train_x, validation_x, test_x, cross_train_x, train_y, validation_y, test_y, cross_train_y]

**Abstract**: The purpose of this repository is to show how classification
work through the use of KNN and SVM and how the results obtained change
with the tuning of the various hyperparameters of the algorithms. To do
that [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html3) was used and some tests are been
conducted:

-   For  *K* in [1,3,5,7] KNN was built.

-   For  *C* in [0.001, 0.01, 0.1, 1, 10, 100, 1000] linear SVM was
    built.

-   For  *C* in [0.001, 0.01, 0.1, 1, 10, 100, 1000] SVM with RBF    kernel was built.

-   A grid search was performed for *gamma* and *C* for each possible
    pair an SVM with RBF kernel was built.

-   A grid search was performed for *gamma* and *C* but this time
    perform 5-fold validation.

For each experiment, the best model was used to evaluate the test set.



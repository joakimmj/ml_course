from sklearn import metrics


def review_clf(clf, test_data: iter, test_labels: iter):
    predictions = clf.predict(test_data)

    classification_report = metrics.classification_report(test_labels, predictions)
    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)
    f1_score = metrics.f1_score(test_labels, predictions)
    precision = metrics.precision_score(test_labels, predictions)
    recall = metrics.recall_score(test_labels, predictions)

    prediction = clf.predict_proba(test_data)

    fpr, tpr, _ = metrics.roc_curve(test_labels, prediction[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    print('Classification report:\n%s\n\nConfusion matrix:\n%s\n\nF-score: %.2f\nPrecision: %.2f\nRecall: %.2f\n'
          'ROC area: %.2f' % (classification_report, confusion_matrix, f1_score, precision, recall, roc_auc))

from help_functions import validate_classifier, validate_regression

if __name__ == '__main__':
    validate_classifier.execute_spam_filter(show=10, cache_data=False, rows=5000)
    print("If you are satisfied with the results, uncomment the guy underneath me to start task 2. (im in main.py :D)")
    #validate_regression.execute_sentiment_analysis(cache_data=False, rows=5000)

from help_functions import validate_classifier

"""
Parameters
----------
show: int
    Decides number of wrongly predicted labels to show (all=-1)
rows: int
    Decides number of data samples to use (all=-1)
"""

if __name__ == '__main__':
    validate_classifier.execute_number_classifier(show=10, rows=5000)

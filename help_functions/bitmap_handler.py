import numpy as np
import matplotlib.pyplot as plt


def __print_bitmap(bitmap: iter, width: int = 28, height: int = 28):
    """
    Plots bitmap.
    :param bitmap: iter
        1d-array that represent the bitmap.
    :param width: int
        Width of the bitmap.
    :param height: int
        Height of the bitmap.
    """
    rows = np.empty(shape=(0, width))

    for h in range(height):
        rows = np.append(rows, [bitmap[h * width: (h + 1) * width]], axis=0)

    plt.imshow(rows)
    plt.gray()
    plt.show()


def compare_wrong_results(bitmaps: iter, prediction: iter, labels: iter):
    """
    Plots list of bitmaps.
    :param bitmaps: iter
        2d-array with bitmaps represented by 1d-arrays.
    :param prediction: iter
        Predicted labels.
    :param labels: iter
        Correct labels.
    """
    for i in range(len(labels)):
        if prediction[i] != labels[i]:
            plt.title("Label: %d, Prediction: %d" % (labels[i], prediction[i]))
            __print_bitmap(bitmaps[i])

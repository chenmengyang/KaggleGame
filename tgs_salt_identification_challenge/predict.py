import numpy as np
import matplotlib.pyplot as plt
from metrices import my_iou_metric1
from skimage.transform import resize

def predict(model, datasetX, datasetY):
    model.load_weights("./data/keras.model")
    def predict_result(model, x_test): # predict both orginal and reflect x
        preds_test = model.predict(x_test).reshape(-1, 128, 128)
        preds_test += np.array([ np.fliplr(a) for a in model.predict(np.array([np.fliplr(x) for x in x_test])).reshape(-1, 128, 128)])
        return preds_test / 2.0
    valP = predict_result(model, datasetX)
    datasetY = datasetY.reshape(-1, 128, 128)
    ths = np.arange(0.4, 1, 0.05)
    ious = [my_iou_metric1(datasetY, 1*(valP>t)) for t in ths]
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = ths[threshold_best_index]
    try:
        plt.plot(ths, ious)
        plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
        plt.xlabel("Threshold")
        plt.ylabel("IoU")
        plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
        plt.legend()
        plt.savefig('./data/figures/val_threshold_bst.png')
    except:
        pass
    testY1 = [resize(timg, (101, 101)) for timg in model.predict(datasetX).reshape(-1, 128, 128)]
    testY1 = np.array([1*(im > threshold_best) for im in testY1])
    return testY1
import orjson
import matplotlib.pyplot as plt

X_AXIS = {
    "switch_5_count": ["0-sw", "1-sw", "2-sw", "3-sw", "4-sw", "5-sw"]
}
Y_LIMS = {
    "unigram_precision": [0.5, 1.1],
    "unigram_recall": [0.5, 1.1]
}

with open("results.json", "rb") as f:
    results = orjson.loads(f.read())

for prediction_method, d in results.items():
    for metric, e in d.items():
        fig, ax = plt.subplots()
        for exp, values in e.items():
            ax.plot(X_AXIS[prediction_method], values, label=exp)
        ax.set_title(f"{metric} on {prediction_method} decoded sequences")
        ax.set_ylim(*Y_LIMS[metric])
        ax.set_xticklabels(X_AXIS[prediction_method], rotation=45)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{prediction_method}.{metric}.png")
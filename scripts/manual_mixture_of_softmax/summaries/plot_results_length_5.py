import orjson
import matplotlib.pyplot as plt

def unzip(l):
    n = len(l[0])
    results = [[] for _ in range(n)]
    for tup in l:
        for i, item in enumerate(tup):
            results[i].append(item)
    return results
X_AXIS = {
    "switch_5_count": ["0-sw", "1-sw", "2-sw", "3-sw", "4-sw"]
}
Y_LIMS = {
    "unigram_precision": [0.5, 1.1],
    "unigram_recall": [0.5, 1.1],
    "length_mismatch": [3.0, -0.1]
}
REVERSE = {
    "unigram_precision": True,
    "unigram_recall": True,
    "length_mismatch": False
}

with open("results_length_5.json", "rb") as f:
    results = orjson.loads(f.read())

for data, res in results.items():
    for prediction_method, d in res.items():
        for metric, e in d.items():
            print(prediction_method, metric)
            fig, ax = plt.subplots(figsize=(10,5))
            ys = []
            for exp, values in e.items():
                if exp == "baseline INTVN":
                    color = "cyan"
                    linewidth = 1
                elif exp == "baseline":
                    color = "blue"
                    linewidth = 1
                else:
                    color= None
                    linewidth = 0.5
                ys.append(values[:5])
                ax.plot(X_AXIS[prediction_method], values[:5], label=exp, linewidth=0.5, color=color)
            ax.set_title(f"{metric} on {prediction_method} decoded sequences")
            ax.set_ylim(*Y_LIMS[metric])
            ax.set_xticklabels(X_AXIS[prediction_method], rotation=45)
            plt.legend(loc="upper left", bbox_to_anchor=(1.04,1))
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            _, labels, handles = unzip(sorted(zip(ys, labels, handles), key=lambda t: t[0][-1], reverse=REVERSE[metric]))
            leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.04,1))
            # leg.set_in_layout( False)
            plt.tight_layout()
            plt.savefig(f"{data}.{prediction_method}.{metric}.len5.png", dpi=500)
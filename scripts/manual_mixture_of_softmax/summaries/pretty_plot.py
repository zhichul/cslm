import orjson
import matplotlib.pyplot as plt

X_AXIS = {
    "switch_5_count": ["0-sw", "1-sw", "2-sw", "3-sw", "4-sw", "5-sw"]
}
Y_LIMS = {
    "unigram_precision": [0.5, 1.1],
    "unigram_recall": [0.5, 1.1]
}


exp = "baseline"
values = []
fig, ax = plt.subplots()
ax.plot(X_AXIS["switch_5_count"], values, label=exp)
ax.set_title(f"{'precision'} on {'switch_5_count'} decoded sequences")
ax.set_ylim(*Y_LIMS['switch_5_count'])
ax.set_xticklabels(X_AXIS["switch_5_count"], rotation=45)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"{'switch_5_count'}.{'precision'}.pretty.png")
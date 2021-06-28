import matplotlib.pyplot as plt
import numpy as np
import orjson


def visualize_matrix(matrix, x_labels, y_labels, **kwargs):
    # pop args
    dpi = kwargs.pop("dpi", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    cmap = kwargs.pop("cmap", None)
    labelsize = kwargs.pop("labelsize", None)
    valuesize = kwargs.pop("valuesize", None)
    title = kwargs.pop("title", None)
    rounding = kwargs.pop("rounding", None)

    # create canvas
    fig, ax = plt.subplots(1, 1, dpi=dpi)

    # visualize
    ax.matshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis='x', which='major', labelsize=labelsize, labelrotation=90)
    ax.tick_params(axis='y', which='major', labelsize=labelsize, labelrotation=0)
    if title: ax.set_title(title)

    # set matrix numbers
    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, str(np.round(z, rounding)), ha='center', va='center', size=valuesize)

    # show
    plt.show()

def visualize_attention(attention_matrix, src_labels, tgt_labels,
                        cmap="gray",
                        labelsize=5,
                        round=2,
                        vsize=5,
                        **kwargs):
    """
    Visualize an attention matrix whose rows sum to 1.
    The x axis corresponds to the attended to,
    and the y axis corresponds to the attended from.
    """

    # In language modeling the labels are offset by one, and the last prediction has no label
    attention_matrix = attention_matrix[:len(tgt_labels)-1, :]
    tgt_labels = tgt_labels[1:]




def visualize_softmix_attention_log(attention_file=None, labels_file=None):
    """
    mixture_prob is head x seq_len
    mixture_attention is head x seq_len x seq_len
    """
    with open(attention_file, "rb") as f:
        data = orjson.loads(f.read())
    with open(labels_file, "rb") as f:
        labels = orjson.loads(f.read())
    for i, (inputs, outputs, mixture_prob, mixture_attention) in enumerate(
            zip(labels["input_tokens"],
                labels["decoder_input_tokens"],
                np.array(data["mixture_probs"]),
                np.array(data["mixture_attentions"]))):
        print(mixture_prob.transpose(1, 0)[:len(outputs) - 1].round(4))
        if "log_probs" in data:
            print(data["log_probs"][i])
        visualize_attention(
            (mixture_attention[:, :, :] * mixture_prob[:, :, None]).sum(axis=0, keepdims=False), inputs, outputs)
        s = input("stop?")
        if s == "y":
            break
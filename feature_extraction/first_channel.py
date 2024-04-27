import numpy as np


def pad_sequences(
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",
        truncating="pre",
        value=0.0,
):
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


def feature_loader1(peptide_seq_dict, peptide_dict):
    peptide_seq_array = np.array(list(peptide_seq_dict.values()))
    peptide_seq_trans = []
    for seq in peptide_seq_array:
        tmp = []
        for i in seq:
            tmp.append(peptide_dict[i])
        peptide_seq_trans.append(tmp)
    maxlen = 0
    for i in peptide_seq_trans:
        if (len(i) > maxlen):
            maxlen = len(i)
    peptide_seq_trans = pad_sequences(peptide_seq_trans, maxlen=maxlen)
    peptide_seq_trans = np.array(peptide_seq_trans)
    return peptide_seq_trans, maxlen

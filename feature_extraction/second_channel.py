import string
from PseudoAAC import GetAAComposition
import numpy as np
from numpy import linalg as la

groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']  # 根据物理化学性质分组
peptide_single = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                  '19', '20']
bpf = []
cksaagp = []
aac = []
rsm = []
dpc = []


def TransDict_from_list(groups):
    transDict = dict()
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group)  # Alphabetically sorted list
        for c in g_members:
            # print('c' + str(c))
            # print('g_members[0]' + str(g_members[0]))
            result[c] = str(tar_list[index])  # K:V map, use group's first letter as represent.
        index = index + 1
    return result


group_dict = TransDict_from_list(groups)


def translate_sequence(seq, translationDict):
    from_list = []  # 原始序列X
    to_list = []  # 翻译序列
    for k, v in translationDict.items():
        from_list.append(k)
        to_list.append(v)
    trans_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    return trans_seq


def BPF(seq_temp):  # 二进制轮廓特征
    seq = seq_temp
    chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    fea = []
    tem_vec = []
    k = 7
    for i in range(k):
        if seq[i] == 'A':
            tem_vec = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'C':
            tem_vec = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'D':
            tem_vec = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'E':
            tem_vec = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'F':
            tem_vec = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'G':
            tem_vec = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'H':
            tem_vec = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'I':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'K':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'L':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'M':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'N':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'P':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'Q':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        elif seq[i] == 'R':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        elif seq[i] == 'S':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        elif seq[i] == 'T':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        elif seq[i] == 'V':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        elif seq[i] == 'W':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        elif seq[i] == 'Y':
            tem_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        fea = fea + tem_vec
    return fea


def CKSAAGP(fastas, gap=5):
    # 计算被任意k个残基隔开的氨基酸对的频率根据物理化学性质组成的不同组别

    def generateGroupPairs(groupKey):
        # CKSAAGP的子函数
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []

    name, sequence = [], fastas
    code = [name]
    for g in range(gap + 1):
        gPair = generateGroupPairs(groupKey)
        sum = 0
        for p1 in range(len(sequence)):
            p2 = p1 + g + 1
            if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                    sequence[p2]]] + 1
                sum = sum + 1

        if sum == 0:
            for gp in gPairIndex:
                code.append(0)
        else:
            for gp in gPairIndex:
                code.append(gPair[gp] / sum)
    code = code[1:]
    return code


def DPC(seq, length, protein_dict):
    matrix = np.zeros((length, length))
    for i in range(len(seq) - 2 + 1):
        matrix[protein_dict[seq[i]]][protein_dict[seq[i + 1]]] += 1
    total_sum = sum(sum(row) for row in matrix)
    normalized_matrix = [[val / total_sum for val in row] for row in matrix]
    flattened_list = []
    for row in normalized_matrix:
        flattened_list.extend(row)
    return flattened_list


def AAC(seq):
    code = GetAAComposition(seq)
    res = []
    for v in code.values():
        res.append(v)
    return res


def translate_single(seq):
    protein_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    seq_trans = [protein_dict[aa] for aa in seq]
    return seq_trans


def RSM(peptide_single, seq_trans):
    seq_len = len(seq_trans)
    single_feature = [0] * len(peptide_single)
    note_feature = [[0 for cols in range(len(seq_trans))] for rows in range(len(peptide_single))]
    for x in range(len(seq_trans)):
        aa = seq_trans[x]
        if str(aa) in peptide_single:
            ind = peptide_single.index(str(aa))
            note_feature[ind][x] = note_feature[ind][x] + 1
    u, s, v = la.svd(note_feature)
    for i in range(len(s)):
        single_feature = single_feature + u[i] * s[i] / seq_len
    return single_feature


def feature_loader2(peptide_seq_dict, peptide_dict):
    for i in peptide_seq_dict:
        protein_seq = translate_sequence(peptide_seq_dict[i], group_dict)
        bpf_feature = BPF(peptide_seq_dict[i])
        aac_feature = AAC(peptide_seq_dict[i])
        cksaagp_feature = CKSAAGP(peptide_seq_dict[i])
        dpc_feature = DPC(peptide_seq_dict[i], 20, peptide_dict)
        seq_trans = translate_single(peptide_seq_dict[i])
        rsm_feature = RSM(peptide_single, seq_trans)
        bpf.append(bpf_feature)
        cksaagp.append(cksaagp_feature)
        aac.append(aac_feature)
        dpc.append(dpc_feature)
        rsm.append(rsm_feature)
    feature2 = np.concatenate(
        (np.array(bpf), np.array(cksaagp), np.array(aac), np.array(dpc), np.array(rsm)), axis=1)
    return feature2

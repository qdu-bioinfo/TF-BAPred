from preprocessing import data_loader
from preprocessing.data_loader import peptide_seq_dict


def reverse(peptide_seq_dict,label):  # 翻转序列
    reversed_peptide_seq_dict = {}
    for key, value in peptide_seq_dict.items():
        reversed_peptide_seq_dict[key] = value[::-1]
    # 删除训练集里长度低于7的序列
    keys_to_remove = []
    # 遍历字典找到需要删除的键
    for key, value in peptide_seq_dict.items():
        if len(value) < 7:
            keys_to_remove.append(key)
    # 删除长度低于7的字符串对应的键
    for key in keys_to_remove:
        del peptide_seq_dict[key]
        del label[key]
    return reversed_peptide_seq_dict,label



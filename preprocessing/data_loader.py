peptide_seq_dict = {}
label = {}
label_index = 0
peptide_index = 0


def get_data():
    label_index = 0
    peptide_index = 0
    with open('bioactive_peptide_sequence', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[-2]
                label_temp = values
                if label_temp == '1':
                    label[label_index] = 1
                else:
                    label[label_index] = 0
                label_index = label_index + 1
            else:
                seq = line[:-1]
                peptide_seq_dict[peptide_index] = seq
                peptide_index = peptide_index + 1

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
    return peptide_seq_dict, list(label.values())

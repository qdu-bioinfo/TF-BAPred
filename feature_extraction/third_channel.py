import numpy as np

fvg = []


def FVG(seq):
    alphabet = {
        0: 'A',
        1: 'C',
        2: 'D',
        3: 'E',
        4: 'F',
        5: 'G',
        6: 'H',
        7: 'I',
        8: 'K',
        9: 'L',
        10: 'M',
        11: 'N',
        12: 'P',
        13: 'Q',
        14: 'R',
        15: 'S',
        16: 'T',
        17: 'V',
        18: 'W',
        19: 'Y'
    }  # 字母表映射
    alphabet = {value: key for key, value in alphabet.items()}
    matrix = [[0] * 40 for _ in range(20)]

    for index in range(len(seq) - 1):
        if index != len(seq) - 1:
            alphabet_index = alphabet[seq[index]]
            next_alphabet_index = alphabet[seq[index + 1]]
            negative_index = alphabet_index * 2 + 1
            next_positive_index = next_alphabet_index * 2
            for i in range(next_alphabet_index + 1):
                matrix[i][negative_index] = matrix[i][negative_index] + 1
                matrix[i][next_positive_index] = matrix[i][next_positive_index] + 1

            if negative_index > next_positive_index:
                for i in range(next_positive_index + 1, negative_index):
                    matrix[next_alphabet_index][i] = matrix[next_alphabet_index][i] + 1
            else:
                for i in range(negative_index + 1, next_positive_index):
                    matrix[next_alphabet_index][i] = matrix[next_alphabet_index][i] + 1

    min_val = min([min(row) for row in matrix])
    max_val = max([max(row) for row in matrix])
    normalized_matrix = [[(val - min_val) / (max_val - min_val) for val in row] for row in matrix]
    flatten_matrix = [element for row in normalized_matrix for element in row]
    return flatten_matrix


def feature_loader3(peptide_seq_dict):
    for i in peptide_seq_dict:
        glob_feature = FVG(peptide_seq_dict[i])
        fvg.append(glob_feature)
    return np.array(fvg)

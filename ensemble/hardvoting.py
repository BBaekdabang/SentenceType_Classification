# -*- coding: utf-8 -*-
"""HardVoting

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_JTpkEQoidkQdufAcF8j-1L-OsKTMbSh
"""

voting_list = [infer1, infer2, infer3]

def HardVoting(voting_list) :

    dic_tmp = {'ID' : [], 'Target' : []}

    for i in range(len(infer)) :
        tmp = []
        for infer in voting_list :
            tmp.append(infer['Target'][i])

        dic_tmp['ID'].append(infer['ID'][i])
        dic_tmp['Target'].append(Counter(tmp).most_common(n=1)[0][0])

    return pd.DataFrame(dic_tmp)
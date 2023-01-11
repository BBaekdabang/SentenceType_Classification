import numpy as np
from tqdm import tqdm

voting_list = [model1, model2, model3]

def SoftVoting(voting_list, test_loader, device):
    for model in voting_list:
        model.to(device)
        model.eval()

    test_predict = []

    for input_ids, attention_mask in tqdm(test_loader):

        input_id = input_ids.to(device)
        mask = attention_mask.to(device)
        tmp = []

        for model in voting_list :
            tmp.append(model(input_id, mask))

        test_predict += (( np.sum(tmp) ) / len(voting_list) ).argmax(1).detach().cpu().numpy().tolist()

    return test_predict

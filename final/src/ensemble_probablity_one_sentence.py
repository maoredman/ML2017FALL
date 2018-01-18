import numpy as np
import pandas as pd
import pickle
import pdb
def main():
    
    name =  ["r-net_Jan-15_12-16.pkl", "r-net_Jan-15_16-27.pkl", 
        "r-net_Jan-15_16-22.pkl", "r-net_Jan-15_12-16_back.pkl"]
    multi_retreive_prob = []
    qid_list = pickle.load(open("qid_list.pkl", "rb"))
    passage_list = pickle.load(open("passage_list.pkl", "rb"))
    offset_list = pickle.load(open("offset_list.pkl", "rb"))
    # load
    for model_name in name:
        multi_retreive_prob.append(pickle.load(open("retreive_prob_" + model_name, "rb")))
    # mean
    N, ensemble_number = len(multi_retreive_prob[0]), len(name)
    mean_retreive_prob = []
    for i in range(N):
        begin, end = 0., 0.
        for j in range(ensemble_number):
            tmp_begin, tmp_end = multi_retreive_prob[j][i]
            begin += tmp_begin
            end += tmp_end
        # mean
        mean_retreive_prob.append((begin/ensemble_number, end/ensemble_number))
    # take argmax and do some fix (like start > end)
    pred_result = []
    for i, (begin, end) in enumerate(mean_retreive_prob):
        
        pred_start = np.argmax(begin, axis=0) # already flatten
        pred_end = np.argmax(end, axis=0) # as above
        if pred_start > pred_end:
            max_value = float("-inf")
            # force pred_end to be bigger than pred_start
            for j in range(pred_start, len(end)):
                if end[j] > max_value:
                    max_value = end[j]
                    pred_end = j
        # restore to actual length of word
        ans_start = sum([len(passage_list[i][j]) for j in range(0, pred_start)])
        ans_start += offset_list[i]
        ans_end = sum([len(passage_list[i][j]) for j in range(0, pred_end+1)])-1
        ans_end += offset_list[i]
        ans_indices = " ".join([str(j) for j in range(ans_start, ans_end+1)])
        qid = qid_list[i]
        pred_result.append((qid, ans_indices))

    df = pd.DataFrame(pred_result, columns=["id", "answer"])
    
    df.to_csv("R_Net_output_0115_2230_ensemble_II", index=False)

if __name__ == '__main__':
    main()
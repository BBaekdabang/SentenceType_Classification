# SentenceType_Classification
# 데이콘 문장 유형 분류 AI 경진대회
<img width="1000" img height="210" alt="Dacon" src="https://user-images.githubusercontent.com/113493692/209252540-79ffd192-36c2-40bb-8d76-30fa54cc6361.png">

#### 데이터셋 다운로드 : [데이콘 문장 유형 분류 AI 경진대회](https://dacon.io/competitions/official/236037/data#)

대회 참여 기간 : 2022.12.12. ~ 2022.12.23.



---

# Code

<table>
    <thead>
        <tr>
            <th>목록</th>
            <th>파일명</th>
            <th>설명</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=1>K-Fold</td>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/k-fold/Stratified%20K-Fold.ipynb">Stratified K-Fold.ipynb</a>
            </td>
            <td> Startified K-Fold </td>
        </tr>
        <tr>
            <td rowspan=3>Train</td>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/BERT/bert.ipynb">bert.ipynb</a>     
            <td> BERT/RoBERTa/ELECTRA </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/T5/t5_train.py">t5_train.py</a>
            <td> T5 </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/MLP/mlp_train.py">mlp_train.py</a>
            <td> MLP </td>
        </tr>
        <tr>
            <td rowspan = 3>Evaluate</td>
            <td>
                <a href="https://github.com/BBaekdabang/EmotionClassification/blob/main/Inference.ipynb">Inference.ipynb</a>     
            <td> BERT/RoBERTa/ELECTRA </td>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/T5/t5_evaluate.py">t5_evaluate.py</a>
            <td> T5 </td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/MLP/mlp_evaluate.py">mlp_evaluate.py</a>
            <td> MLP </td>
        </tr>
        </tr>        
        <tr>
            <td rowspan = 2>Ensemble</td>       
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/hardvoting.py">hardvoting.py</a>
            <td> Hard Voting</td>
        </tr>
        <tr>
            <td>
                <a href="https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/softvoting.py">softvoting.py</a>
            <td> Soft Voting</td>
        </tr>

   </tbody>
</table>


---

# 가. 개발 환경

     Google Colab Pro
     
     huggingface-hub==0.10.1
     datasets==2.6.1
     tokenizers==0.13.2
     torch==1.12.1+cu113
     torchvision==0.13.1+cu113
     transformers==4.24.0
     tqdm==4.64.1
     scikit-learn==1.0.2
     sentencepiece==0.1.97


---

# 나. 데이터 예시 
 출처 : Dacon, 문장 유형 분류 AI 경진대회 

    {"ID": "TRAIN-00000", "문장": "0.75%포인트 금리 인상은 1994년 이후 28년 만에 처음이다." "유형": "사실형", "극성": "긍정", "시제": "현재", "확실성" : "확실", "label" : "사실형-긍정-현재-확실"}
    {"ID": "TRAIN-00001", "문장": "이어 ＂앞으로 전문가들과 함께 4주 단위로 상황을 재평가할 예정＂이라며 ＂그 이전이라도 방역 지표가 기준을 충족하면 확진자 격리의무 조정 여부를 검토할 것＂이라고 전했다." "유형": "사실형", "극성": "긍정", "시제": "과거", "확실성" : "확실", "label" : "사실형-긍정-과거-확실"}
    {"ID": "TRAIN-00002", "문장": "정부가 고유가 대응을 위해 7월부터 연말까지 유류세 인하 폭을 30%에서 37%까지 확대한다." "유형": "사실형", "극성": "긍정", "시제": "과거", "확실성" : "확실", "label" : "사실형-긍정-과거-확실"}

---


# 라. 주요 소스 코드

- ## Model Load From Hugging Face
   
   
    <table>
    <thead>
        <tr>
            <th>Model</th>
            <th>링크(HuggingFace)</td>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td> RoBERTa</td>
            <td>
                <a href="https://huggingface.co/klue/roberta-base">klue/roberta-base</a>
        </tr>
        <tr>
            <td> T5</td>            
            <td>
                <a href="https://huggingface.co/paust/pko-t5-base">paust/pko-t5-base</a>
        </tr>
    </tbody>
    </table>
      
    
   ```c
    # HuggingFace에서 불러오기
    from transformers import AutoTokenizer, AutoModel
    base_model = "HuggingFace주소"

    Model = AutoModel.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
   ```


- ## Stratified K-Fold
   
   > [Stratified_K-Fold.ipynb 참조](https://github.com/BBaekdabang/SentenceType_Classification/blob/main/k-fold/Stratified%20K-Fold.ipynb)

     ```c

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    folds=[]

    for train_idx, valid_idx in skf.split(traindata_type, traindata_type['label']):
        train_idx = np.array(list(set(list(train_idx))))
        valid_idx = np.array(list(set(set(valid_idx))))
        folds.append((train_idx, valid_idx))
     ```
    
    
- ## Ensemble
    - #### Hard Voting
        > [hardvoting.py 참조](https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/hardvoting.py)
        
    ```c

    def HardVoting(voting_list) :

        dic_tmp = {'ID' : [], 'Target' : []}

        for i in range(len(infer)) :
            tmp = []
            for infer in voting_list :
                tmp.append(infer['Target'][i])

            dic_tmp['ID'].append(infer['ID'][i])
            dic_tmp['Target'].append(Counter(tmp).most_common(n=1)[0][0])

        return pd.DataFrame(dic_tmp)
    ```
        
    - #### Soft Voting        
        > [softvoting.py 참조](https://github.com/BBaekdabang/SentenceType_Classification/blob/main/ensemble/softvoting.py)

    ```c
    
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
    ``` 


---

# 마. Reference

[1] [BERT: Bidirectional Encoder Representations from Transformers](https://arxiv.org/pdf/1810.04805v2.pdf) : Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).

[2] [ELECTRA : Pre-training text encoders as discriminators rather than generators](https://arxiv.org/pdf/2003.10555.pdf%3C/p%3E) : Clark, Kevin, et al. "Electra: Pre-training text encoders as discriminators rather than generators." arXiv preprint arXiv:2003.10555 (2020).

[3] [RoBERTa : A robustly optimized bert pretraining approach](https://arxiv.org/pdf/1907.11692.pdf%5C) : Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).

[4] [T5 : Text-to-Text Transfer Transformer](https://arxiv.org/pdf/1910.10683v3.pdf) : Bujard, Hermann, et al. "[26] A T5 promoter-based transcription-translation system for the analysis of proteins in vitro and in vivo." Methods in enzymology. Vol. 155. Academic Press, 1987. 416-433.

---

# 바. Members
Hyoje Jung | flash1253@naver.com<br>
Yongjae Kim | dydwo322@naver.com<br>
Hyein Oh | gpdls741@naver.com<br>
Seungyong Guk | kuksy77@naver.com<br>
Jaehyeog Lee | tysl4545@naver.com<br>
Hyojin Kang | khj94111@gmail.com

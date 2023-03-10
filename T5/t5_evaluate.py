from transformers import AutoTokenizer

def predict(tokenizer, model, data):
    
    global label_list

    for i in range(len(data)):

        form = preprocessing(data['문장'][i])
        data['정답'][i] = []
        
        tokenized_data = tokenizer(form, padding='max_length', max_length=256, truncation=True)

        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        outputs = model.model_PLM.generate(
            input_ids=input_ids,
            attention_mask=attention_mask)
        pred_t = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
        data['정답'][i] = pred_t[0]

    return data

def test():

    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-base')
     
    test_data = pd.read_csv( test_data_path )
            
    model = T5_model(len(label_list), len(tokenizer))
    model.load_state_dict(torch.load( model_path , map_location=device ))
    model.to(device)
    model.eval()
        
    pred_data = predict(tokenizer, model, copy.deepcopy(test_data))
    pred_data.to_csv('', encoding = 'utf-8', index = False)

    return pd.DataFrame(pred_data)

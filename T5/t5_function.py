def preprocessing(sentence) :

    if category == '유형' :
        return "문장의 유형을 분류하시오: " + sentence

    if category == '극성' :
        return "문장의 극성을 판단하시오: " + sentence

    if category == '시제' :
        return "문장의 시제를 찾으시오: " + sentence

    if category == '확실성' :
        return "문장의 확실성을 판단하시오: " + sentence

def tokenize_and_align_labels(tokenizer, form, annotations):

    entity_encode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    entity_decode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    
    answer_label = "<pad>"
    sentence = preprocessing(form)
    tokenized_data = tokenizer(sentence, padding='max_length', max_length=256, truncation=True)

    answer_label = answer_label + annotations + " "

    tokenized_label = tokenizer(answer_label[:-1], padding='max_length', max_length=20, truncation=True)
    
    entity_encode_data_dict['input_ids'].append(tokenized_data['input_ids'])
    entity_encode_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    
    entity_decode_data_dict['input_ids'].append(tokenized_label['input_ids'])
    entity_decode_data_dict['attention_mask'].append(tokenized_label['attention_mask'])        

    return entity_encode_data_dict, entity_decode_data_dict

def get_dataset(data_path, tokenizer):
    raw_data = pd.read_csv(data_path)
    input_ids_list = []
    attention_mask_list = []

    decode_input_ids_list = []
    decode_attention_mask_list = []

    for i in range(len(raw_data)):
        
        entity_encode_data_dict, entity_decode_data_dict = tokenize_and_align_labels(tokenizer, raw_data['문장'][i], raw_data['정답'][i])
        input_ids_list.extend(entity_encode_data_dict['input_ids'])
        attention_mask_list.extend(entity_encode_data_dict['attention_mask'])

        decode_input_ids_list.extend(entity_decode_data_dict['input_ids'])
        decode_attention_mask_list.extend(entity_decode_data_dict['attention_mask'])


    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list), torch.tensor(decode_input_ids_list), torch.tensor(decode_attention_mask_list))

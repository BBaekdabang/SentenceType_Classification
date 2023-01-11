# 1. 문장(Text) 벡터화 -> TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 4, analyzer = 'word', ngram_range=(1, 2))
vectorizer.fit(np.array(train["문장"]))

train_vec = vectorizer.transform(train["문장"])
val_vec = vectorizer.transform(val["문장"])
test_vec = vectorizer.transform(test["문장"])

# 2. Label Encoding (유형, 극성, 시제, 확실성)
type_le = preprocessing.LabelEncoder()
train["유형"] = type_le.fit_transform(train["유형"].values)
val["유형"] = type_le.transform(val["유형"].values)

polarity_le = preprocessing.LabelEncoder()
train["극성"] = polarity_le.fit_transform(train["극성"].values)
val["극성"] = polarity_le.transform(val["극성"].values)

tense_le = preprocessing.LabelEncoder()
train["시제"] = tense_le.fit_transform(train["시제"].values)
val["시제"] = tense_le.transform(val["시제"].values)

certainty_le = preprocessing.LabelEncoder()
train["확실성"] = certainty_le.fit_transform(train["확실성"].values)
val["확실성"] = certainty_le.transform(val["확실성"].values)

train_type = train["유형"].values # sentence type
train_polarity = train["극성"].values # sentence polarity
train_tense = train["시제"].values # sentence tense
train_certainty = train["확실성"].values # sentence certainty

train_labels = {
    'type' : train_type,
    'polarity' : train_polarity,
    'tense' : train_tense,
    'certainty' : train_certainty
}

val_type = val["유형"].values # sentence type
val_polarity = val["극성"].values # sentence polarity
val_tense = val["시제"].values # sentence tense
val_certainty = val["확실성"].values # sentence certainty

val_labels = {
    'type' : val_type,
    'polarity' : val_polarity,
    'tense' : val_tense,
    'certainty' : val_certainty
}

class CustomDataset(Dataset):
    def __init__(self, st_vec, st_labels):
        self.st_vec = st_vec
        self.st_labels = st_labels

    def __getitem__(self, index):
        st_vector = torch.FloatTensor(self.st_vec[index].toarray()).squeeze(0)
        if self.st_labels is not None:
            st_type = self.st_labels['type'][index]
            st_polarity = self.st_labels['polarity'][index]
            st_tense = self.st_labels['tense'][index]
            st_certainty = self.st_labels['certainty'][index]
            return st_vector, st_type, st_polarity, st_tense, st_certainty
        else:
            return st_vector

    def __len__(self):
        return len(self.st_vec.toarray())

train_dataset = CustomDataset(train_vec, train_labels)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_vec, val_labels)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

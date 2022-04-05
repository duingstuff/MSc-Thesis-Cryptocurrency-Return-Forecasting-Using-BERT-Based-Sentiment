# experiment with financial phrasebank data (Malo et al.) for the weak labeling approach

# load data from Malo et al.
data = pd.DataFrame(open("data/Sentences_50Agree.txt", "r"))

data = data[0].apply(lambda x: x.rsplit('@', 1))

# split data
data_split = pd.DataFrame()
data_split['text'] = data.apply(lambda x: x[0])
data_split['label'] = data.apply(lambda x: x[1])
data_split['label'] = data_split['label'].apply(lambda x: x.replace("\n", ""))

train, test = train_test_split(data_split, test_size=0.2, random_state=0)
train, valid = train_test_split(train, test_size=0.2, random_state=0)

# zero-shot classifier output for the model (use other scripts to compute the zsc output)
zsc_org_output = pd.read_excel("zsc_org_output_50agree_posnegonly.xlsx", 
                               index_col=0).dropna().drop_duplicates().reset_index(drop=True)
zsc_org_output['prediction'].loc[zsc_org_output['score'] < 0.90] = "neutral"

train = zsc_org_output.loc[zsc_org_output['text'].isin(train['text'])].reset_index(drop=True)
valid = zsc_org_output.loc[zsc_org_output['text'].isin(valid['text'])].reset_index(drop=True)
test = zsc_org_output.loc[zsc_org_output['text'].isin(test['text'])].reset_index(drop=True)

polarity_ratios(train, ['target'])
polarity_ratios(valid, ['prediction'])
polarity_ratios(test, ['prediction'])

train.to_csv('train_data.csv', sep='\t')
valid.to_csv('valid_data.csv', sep='\t')
test.to_csv('test_data.csv', sep='\t')

np.unique(test.prediction)

# fit bert using weak labels ----

# define tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# encode all data
train_inputs, train_masks = encode_all(train['text'], tokenizer)
validation_inputs, validation_masks = encode_all(valid['text'], tokenizer)

# labels of the data
train_labels = find_label(train.prediction.values)
validation_labels = find_label(valid.prediction.values)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# define model as pre-trained BERT with classification layer
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 3,   
    output_attentions = False, 
    output_hidden_states = False)

model.parameters
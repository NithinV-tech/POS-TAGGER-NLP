#####################################################################################################3
import numpy as np
import torch
import conllu
from collections import Counter
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import Adam
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
####################################################################################################
####################################DATA PRE PROCESSING########################################################################
def parse_conllu(file_path):
    sentences = []
    word_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        parsed_data = conllu.parse(data)

        for sentence in parsed_data:
            sentence_words = [(token['form'].lower(), token['upostag'])
                              for token in sentence if token['upostag'] is not None and token['form'].isalnum()]
            word_counts.update([word for word, _ in sentence_words])
            sentences.append(sentence_words)

    return sentences, word_counts

def replace_oov(sentences, word_counts):
    oov_threshold = 3
    for sentence in sentences:
        for i, (word, pos_tag) in enumerate(sentence):
            if word_counts[word] < oov_threshold:
                sentence[i] = ('OOV', pos_tag)

    return sentences

train_file_path = 'en_atis-ud-train.conllu'
train_sentences, train_word_counts = parse_conllu(train_file_path)
train_sentences_oov = replace_oov(train_sentences, train_word_counts)
#print(train_sentences_oov)

def load_glove_embeddings(glove_path):
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings

glove_path = 'glove.6B.100d.txt'
glove_embeddings = load_glove_embeddings(glove_path)

def word_to_embedding(sentences, glove_embeddings):
    embedding_dim = 100  
    for sentence in sentences:
        for i, (word, pos_tag) in enumerate(sentence):
            if word in glove_embeddings:
                sentence[i] = (glove_embeddings[word], pos_tag)
            else:
              
                sentence[i] = (np.zeros(embedding_dim), pos_tag)
    return sentences

train_sentences_embedded = word_to_embedding(train_sentences_oov, glove_embeddings)
'''
pos_tag_set = set()
for sentence in train_sentences_embedded:
    for _, pos_tag in sentence:
        pos_tag_set.add(pos_tag)

pos_tag_to_index = {tag: idx for idx, tag in enumerate(sorted(pos_tag_set))}
'''
#################################################### end of pre processing#####################################################3

#################################################INITIAL MODEL###################################################################

class POSDataset(Dataset):
    def __init__(self, embedded_sentences, tag_to_ix):
        self.embedded_sentences = embedded_sentences
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.embedded_sentences)

    def __getitem__(self, idx):
        sentence, tags = zip(*self.embedded_sentences[idx])
        sentence_tensor = torch.FloatTensor(sentence) 
        tag_indices = torch.LongTensor([self.tag_to_ix[tag] for tag in tags])
        return sentence_tensor, tag_indices

dev_file_path = 'en_atis-ud-dev.conllu'
dev_sentences, dev_word_counts = parse_conllu(dev_file_path)
dev_sentences_oov = replace_oov(dev_sentences, dev_word_counts)
dev_sentences_embedded = word_to_embedding(dev_sentences_oov, glove_embeddings)

st = set()
for sentence in train_sentences_embedded:
  for _, tag in sentence:
    st.add(tag)

for sentence in dev_sentences_embedded:
  for _, tag in sentence:
    st.add(tag)
all_pos_tags = list(st)
tag_to_ix = {tag: ix for ix, tag in enumerate(all_pos_tags)}
dataset = POSDataset(train_sentences_embedded, tag_to_ix)
#print(dataset)
dv_dataset = POSDataset(dev_sentences_embedded, tag_to_ix)

def collate_fn(batch):
    sentences, tags = zip(*batch)
    #zero_embedding = np.zeros((1, embedding_dim), dtype=np.float32)  
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    #max_tag_index = max([tag.max().item() for tag in tags])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=len(tag_to_ix))
    return sentences_padded, tags_padded

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dv_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
#print(tag_to_ix)



class RNNPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers=2, bidirectional=True, activation_fn=nn.ReLU()):
        super(RNNPOSTagger, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.activation_fn = activation_fn
        self.hidden2tag = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        lstm_out = self.activation_fn(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores

embedding_dim = 100 
hidden_dim = 128  
output_dim = len(tag_to_ix)+1
model = RNNPOSTagger(embedding_dim, hidden_dim, output_dim)
ignore_index = 14
loss_function = nn.NLLLoss()
optimizer = Adam(model.parameters())
for epoch in range(3):
    total_loss = 0
    for sentences, tags in dataloader:
        model.zero_grad()
        tag_scores = model(sentences)
        flat_tag_scores = tag_scores.view(-1, output_dim)
        flat_tags = tags.view(-1)
        non_pad_indices = (flat_tags != len(tag_to_ix)).nonzero().squeeze()
        filtered_tag_scores = flat_tag_scores[non_pad_indices]
        filtered_tags = flat_tags[non_pad_indices]
        loss = loss_function(filtered_tag_scores, filtered_tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
#print("TRAINING COMPLETED!!!!!!!!!!!!!!!!!!!!!!!!!")
#print("#############################################################")
    
#####################3PREDICTION#################################################################################3333

def predict_tags(model, sentence, glove_embeddings, tag_to_ix):
    tokens = sentence.split()
    embedded_sentence = [(glove_embeddings.get(word, np.zeros(100)), 'OOV') for word in tokens]
    embedded_tensor = torch.FloatTensor([embedding for embedding, _ in embedded_sentence])
    with torch.no_grad():
        tag_scores = model(embedded_tensor.unsqueeze(0)) 
        _, predicted_indices = torch.max(tag_scores, dim=2)
    predicted_tags = [list(tag_to_ix.keys())[idx] for idx in predicted_indices.squeeze().tolist()]
    return list(zip(tokens, predicted_tags))
sentence = input("Enter a sentence: ")
predicted_tags = predict_tags(model, sentence, glove_embeddings, tag_to_ix)
print(predicted_tags)
############################################################################################################################

###################################HYPER PARAMETER TUNING###############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyperparameters = [
    {'num_layers': 1, 'hidden_dim': 128, 'bidirectional': True, 'activation_fn': nn.ReLU()},
    {'num_layers': 2, 'hidden_dim': 256, 'bidirectional': False, 'activation_fn': nn.Tanh()},
    {'num_layers': 2, 'hidden_dim': 128, 'bidirectional': True, 'activation_fn': nn.LeakyReLU()}
]


def evaluate_model(model, dataloader, device):
    model.eval() 
    total_correct = 0
    total_tokens = 0

    with torch.no_grad(): 
        for sentences, tags in dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            output = model(sentences)
            _, predicted_tags = torch.max(output, dim=2) 
            valid_indices = tags !=len(tag_to_ix)
            total_correct += (predicted_tags[valid_indices] == tags[valid_indices]).sum().item()
            total_tokens += valid_indices.sum().item()
    accuracy = (total_correct / total_tokens) * 100
    return accuracy


def train_and_evaluate_model_for_tuning(config, train_dataloader, dev_dataloader, device):
    model_tune = RNNPOSTagger(
        embedding_dim=100,
        hidden_dim=config['hidden_dim'],
        output_dim=len(tag_to_ix) + 1,
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        activation_fn=config['activation_fn']
    ).to(device)

    optimizer = Adam(model_tune.parameters())  
    loss_function = nn.NLLLoss(ignore_index=len(tag_to_ix)) 
    epoch_accuracies = []
    for epoch in range(5):
        model_tune.train()
        total_loss = 0
        for sentences, tags in train_dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            model_tune.zero_grad()
            tag_scores = model_tune(sentences)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix) + 1), tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f'Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}')

        dev_accuracy = evaluate_model(model_tune, dev_dataloader, device) 
        epoch_accuracies.append(dev_accuracy)
    return  model_tune,dev_accuracy,epoch_accuracies

best_model = None
best_accuracy = 0
best_config = None

config_accuracies = {}
for config in hyperparameters:
    model_tune, dev_accuracy,epoch_accuracy = train_and_evaluate_model_for_tuning(config, dataloader, dev_dataloader, device)
    config_accuracies[str(config)] = epoch_accuracy
    if dev_accuracy > best_accuracy:
        best_accuracy = dev_accuracy
        best_model = model_tune
        best_config = config
    print(f"Config: {config}, Dev Accuracy: {dev_accuracy}%")

print(f"Best Config: {best_config}, Best Dev Accuracy: {best_accuracy}%")
torch.save(best_model, 'LSTM_BEST_Model_weights.pt')
print("POSModel weights saved.")


plt.figure(figsize=(10, 7))
for config, accuracies in config_accuracies.items():
    plt.plot(accuracies, label=f"Config: {config}")
plt.xlabel('Epoch')
plt.ylabel('Dev Set Accuracy (%)')
plt.title('Epoch vs. Dev Set Accuracy for Different Configurations')
plt.legend(loc='lower right')
plt.show()

#################################################################################################################3

#########################################TEST SET  EVALUATION#########################################################################
test_file_path = 'en_atis-ud-test.conllu'
test_sentences, test_word_counts = parse_conllu(test_file_path)
test_sentences_oov = replace_oov(test_sentences, test_word_counts)
test_sentences_embedded = word_to_embedding(test_sentences_oov, glove_embeddings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = POSDataset(test_sentences_embedded, tag_to_ix)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_result = evaluate_model(best_model, test_dataloader, device)
print(f" Test Accuracy: {test_result}%")

def evaluate_model_extended(model, dataloader, device, tag_to_ix):
    model.eval()  
    true_tags_all = []
    pred_tags_all = []

    with torch.no_grad():
        for sentences, tags in dataloader:
            sentences = sentences.to(device)
            tags = tags.to(device)
            output = model(sentences)
            _, predicted_tags = torch.max(output, dim=2)
            tags = tags.cpu().numpy().flatten()
            predicted_tags = predicted_tags.cpu().numpy().flatten()
            valid_indices = tags != len(tag_to_ix)
            true_tags = tags[valid_indices]
            pred_tags = predicted_tags[valid_indices]
            true_tags_all.extend(true_tags)
            pred_tags_all.extend(pred_tags)

    labels = np.arange(len(tag_to_ix) + 1) 
    accuracy = accuracy_score(true_tags_all, pred_tags_all)
    recall_micro = recall_score(true_tags_all, pred_tags_all, average='micro', labels=labels)
    recall_macro = recall_score(true_tags_all, pred_tags_all, average='macro', labels=labels)
    f1_micro = f1_score(true_tags_all, pred_tags_all, average='micro', labels=labels)
    f1_macro = f1_score(true_tags_all, pred_tags_all, average='macro', labels=labels)
    cm = confusion_matrix(true_tags_all, pred_tags_all, labels=labels)
    display_labels = [tag for tag, ix in sorted(tag_to_ix.items(), key=lambda x: x[1])] + ["PAD"]
    return accuracy, recall_micro, recall_macro, f1_micro, f1_macro, cm, display_labels


def plot_confusion_matrix(cm, display_labels, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='vertical')
    plt.title(title)
    plt.show()


ctr=0

def print_evaluation_metrics(metrics, dataset_name):
    global ctr
    accuracy, recall_micro, recall_macro, f1_micro, f1_macro = metrics
    print(f"{dataset_name} Set Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Recall (Micro): {recall_micro * 100:.2f}%")
    print(f"Recall (Macro): {recall_macro * 100:.2f}%")
    print(f"F1 Score (Micro): {f1_micro * 100:.2f}%")
    print(f"F1 Score (Macro): {f1_macro * 100:.2f}%")
    print("Confusion Matrix:")
    if ctr==0:
        plot_confusion_matrix(metrics_dev[5], metrics_dev[6], title='Confusion Matrix - Dev Set')
        ctr+=1
    else:
        plot_confusion_matrix(metrics_test[5], metrics_test[6], title='Confusion Matrix - Test Set')
    print("\n")

   # print("Classes:")
    #print(classes)
    #anprint("\n")


metrics_dev = evaluate_model_extended(best_model, dev_dataloader, device, tag_to_ix)
print("######################################################################")
print_evaluation_metrics(metrics_dev, "Dev set accuracy parameters")
metrics_test = evaluate_model_extended(best_model, test_dataloader, device, tag_to_ix)
print("######################################################################")
print_evaluation_metrics(metrics_test, "Test set accuracy parameters")

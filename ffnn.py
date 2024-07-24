########################################LOADERS######################################################################3
import numpy as np
import torch
import conllu
from collections import Counter
import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
############################################################################################################################


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

OOV_VECTOR = np.random.rand(100)
train_file_path = 'en_atis-ud-train.conllu'
train_sentences, train_word_counts = parse_conllu(train_file_path)
train_sentences_oov = replace_oov(train_sentences, train_word_counts)
#print(train_sentences_oov)
########################dev sentence###########################33
dev_file_path = 'en_atis-ud-dev.conllu'
dev_sentences, dev_word_counts = parse_conllu(dev_file_path)
dev_sentences_oov = replace_oov(dev_sentences, dev_word_counts)
####################################################################
def load_glove_embeddings(glove_path):
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings
glove_path = './glove.6B.100d.txt' 
#glove_path = 'https://drive.google.com/uc?id=1R2_AE4lfgn2DsW9sGm2yuImw-6MZNqaG'
glove_embeddings = load_glove_embeddings(glove_path)

#########################################################################3
def word_to_embedding(sentences, glove_embeddings):
    embedding_dim = 100  
    for sentence in sentences:
        for i, (word, pos_tag) in enumerate(sentence):
            if word in glove_embeddings:
                sentence[i] = (glove_embeddings[word], pos_tag)
            else:
                sentence[i] = (np.zeros(embedding_dim), pos_tag)
                #sentence[i] = (OOV_VECTOR, pos_tag)
    return sentences
train_sentences_embedded = word_to_embedding(train_sentences_oov, glove_embeddings)
dev_sentences_embedded =  word_to_embedding(dev_sentences_oov, glove_embeddings)

###############################################################################################33p
embedding_dim = 100
st = set()
for sentence in train_sentences_embedded:
  for _, tag in sentence:
    st.add(tag)

for sentence in dev_sentences_embedded:
  for _, tag in sentence:
    st.add(tag)
unique_pos_tags1 = list(st)
pos_to_index= {tag: ix for ix, tag in enumerate(unique_pos_tags1)}
#print(pos_to_index)


def create_input_vectors(sentences, p, s):
    X = [] 
    Y = []  
    for sentence in sentences:
        embeddings = [word[0] for word in sentence] 
        pos_tags = [word[1] for word in sentence]
        embeddings = [np.zeros(embedding_dim)] * p + embeddings + [np.zeros(embedding_dim)] * s
        for i in range(p, len(sentence) - s):
            context_embeddings = embeddings[i-p:i+s+1] 
            input_vector = np.concatenate(context_embeddings)
            X.append(input_vector)
            Y.append(pos_to_index[pos_tags[i-p]])   
    return np.array(X), np.array(Y)
X,Y = create_input_vectors(train_sentences_embedded,2,3)
X_Dev,Y_Dev = create_input_vectors(dev_sentences_embedded,2,3)

##############################################end of pre processing##################################################################################

#############################################INITIAL MODEL#########################################################################3

class POSModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(POSModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1) 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x) 
        x = self.fc2(x)
        x = self.softmax(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.long).to(device)
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
input_dim = X.shape[1] 
hidden_dim = 128  
output_dim = len(pos_to_index)
model = POSModel(input_dim, hidden_dim, output_dim)
criterion = nn.NLLLoss() 
optimizer = optim.Adam(model.parameters())
num_epochs = 5 
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)     
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')
 

#torch.save(model_info, 'POSModel_weights.pt')
#print("POSModel weights saved.")

'''
def load_POSModel(input_dim, hidden_dim, output_dim,device):
    model = POSModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('POSModel_weights.pt',map_location=device))
    model.to(device)
    #state_dict = torch.load('POSModel_weights.pt')
    #print(state_dict.keys()) 

    #criterion = nn.NLLLoss() 
    #optimizer = optim.Adam(model.parameters())
    model.eval()
    return model

loaded_POSModel = load_POSModel(input_dim, hidden_dim, output_dim,device)

'''
###############################################PREDICTION PART ###################################################################################
def process_input_sentence(sentence, glove_embeddings, p, s, embedding_dim=100):
    tokens = sentence.lower().split()
    embeddings = [glove_embeddings.get(token, np.zeros(embedding_dim)) for token in tokens]
    padded_embeddings = [np.zeros(embedding_dim)] * p + embeddings + [np.zeros(embedding_dim)] * s
    X = []
    for i in range(p, len(padded_embeddings) - s):
        context_embeddings = padded_embeddings[i-p:i+s+1] 
        input_vector = np.concatenate(context_embeddings)
        X.append(input_vector)
    return np.array(X), tokens

def predict_pos_tags(model1, sentence, glove_embeddings, pos_to_index, p=2, s=3):
    X, tokens = process_input_sentence(sentence, glove_embeddings, p, s)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model1.eval()
    with torch.no_grad():
        outputs = model1(X_tensor)
        _, predicted_indices = torch.max(outputs, 1)
    index_to_pos = {v: k for k, v in pos_to_index.items()}
    predicted_pos_tags = [index_to_pos[index.item()] for index in predicted_indices]
    return list(zip(tokens, predicted_pos_tags))

sentence = input("Enter a sentence: ")
predicted_tags = predict_pos_tags(model, sentence, glove_embeddings, pos_to_index)
for token, pos_tag in predicted_tags:
    print(f'{token} {pos_tag}')

#####################################################################################################################################
    


###########################################HYPER PARAMETER TUNING##################################################################3    
class FlexiblePOSModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU):
        super(FlexiblePOSModel, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(activation_fn())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LogSoftmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train() 
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

def evaluate(model, dataloader):
    model.eval()  
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    accuracy = (correct_predictions / total_predictions)*100
    return {'accuracy': accuracy}

X_dev_tensor = torch.tensor(X_Dev, dtype=torch.float32)
Y_dev_tensor = torch.tensor(Y_Dev, dtype=torch.long)
dev_dataset = TensorDataset(X_dev_tensor, Y_dev_tensor)
dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
############################ done to use in the end confusion matrix#########################33
dev_dataloader2 = dev_dataloader
###############################################################################################################
input_dim = X.shape[1] 
output_dim = len(pos_to_index)
config_metrics = []
models = []
configs = [
    {'hidden_dims': [128], 'activation_fn': nn.ReLU},
    {'hidden_dims': [256, 128], 'activation_fn': nn.LeakyReLU},
    {'hidden_dims': [128, 128, 128], 'activation_fn': nn.Tanh},
]

for config in configs:
    model = FlexiblePOSModel(input_dim, config['hidden_dims'], output_dim, config['activation_fn'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, dataloader, criterion, optimizer, num_epochs=3)
    dev_metrics = evaluate(model, dev_dataloader)
    print(f"Configuration: {config}, Dev Accuracy: {dev_metrics['accuracy']}")
    models.append(model)
    config_metrics.append((config, dev_metrics))

best_index, best_performance = max(enumerate(config_metrics), key=lambda x: x[1][1]['accuracy'])
best_model = models[best_index]
best_config = best_performance[0]
best_dev_metrics = best_performance[1]
print("Best Configuration:", best_config)
print("Best Dev Set Metrics:", best_dev_metrics)
#####################################################################################################################3

################################### context window vs dev-set accuracy####################################################333

context_windows = range(5)
dev_accuracies = []

for context_window in context_windows:
    X_train, Y_train = create_input_vectors(train_sentences_embedded, context_window, context_window)
    X_dev, Y_dev = create_input_vectors(dev_sentences_embedded, context_window, context_window)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    Y_dev_tensor = torch.tensor(Y_dev, dtype=torch.long)
    dev_dataset = TensorDataset(X_dev_tensor, Y_dev_tensor)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    model = FlexiblePOSModel(input_dim=X_train.shape[1], hidden_dims=best_config['hidden_dims'], output_dim=len(pos_to_index), activation_fn=best_config['activation_fn'])
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    train(model, train_dataloader, criterion, optimizer, num_epochs=5)
    dev_metrics = evaluate(model, dev_dataloader)
    dev_accuracies.append(dev_metrics['accuracy'])
plt.figure(figsize=(10, 6))
plt.plot(list(context_windows), dev_accuracies, marker='o', linestyle='-', color='b')
plt.title('Context Window Size vs. Development Set Accuracy')
plt.xlabel('Context Window Size')
plt.ylabel('Development Set Accuracy')
plt.xticks(list(context_windows))
plt.grid(True)
plt.show()
#################################################################################################################3


###############################################TEST SET CHECKING##################################################################
test_file_path = 'en_atis-ud-test.conllu'
test_sentences, _ = parse_conllu(test_file_path)
test_sentences_embedded = word_to_embedding(test_sentences, glove_embeddings)
p = 2
s = 3
X_test, Y_test = create_input_vectors(test_sentences_embedded, p, s)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=64)
test_accuracy = evaluate(best_model, test_dataloader)['accuracy']
print(f'Test Accuracy: {test_accuracy :.2f}%')
######################################################################################################3

########################################EVALUATION METRICS########################################################

def evaluate_extended(model, dataloader, index_to_pos):
    model.eval()  
    true_labels = []
    predictions = []

    with torch.no_grad(): 
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs, 1)
            predictions.extend(predicted_indices.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = [index_to_pos[index] for index in predictions]
    true_labels = [index_to_pos[index] for index in true_labels]
    accuracy = accuracy_score(true_labels, predictions) * 100
    recall_macro = recall_score(true_labels, predictions, average='macro') * 100
    f1_macro = f1_score(true_labels, predictions, average='macro') * 100
    recall_micro = recall_score(true_labels, predictions, average='micro') * 100
    f1_micro = f1_score(true_labels, predictions, average='micro') * 100
    unique_labels = sorted(set(true_labels + predictions))
    cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
    return {
        'accuracy': accuracy,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'labels': unique_labels
    }
index_to_pos = {index: pos for pos, index in pos_to_index.items()}
results = evaluate_extended(best_model, test_dataloader, index_to_pos)
results2 = evaluate_extended(best_model, dev_dataloader2, index_to_pos)
print("#####################################################")
print(" THE  TEST SET EVALUATION MATRICES ")
print(f"Accuracy: {results['accuracy']}%")
print(f"Macro Recall: {results['recall_macro']}%")
print(f"Macro F1: {results['f1_macro']}%")
print(f"Micro Recall: {results['recall_micro']}%")
print(f"Micro F1: {results['f1_micro']}%")
print("####################################################")
print(" THE  DEV SET EVALUATION MATRICES ")
print(f"Accuracy: {results2['accuracy']}%")
print(f"Macro Recall: {results2['recall_macro']}%")
print(f"Macro F1: {results2['f1_macro']}%")
print(f"Micro Recall: {results2['recall_micro']}%")
print(f"Micro F1: {results2['f1_micro']}%")
print("####################################################")

def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)  
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 10}, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout() 
    plt.show()

plot_confusion_matrix(results['confusion_matrix'], results['labels'], title='Confusion Matrix - Test Set')
plot_confusion_matrix(results2['confusion_matrix'], results2['labels'], title='Confusion Matrix - Dev Set')
###########################################################################################################
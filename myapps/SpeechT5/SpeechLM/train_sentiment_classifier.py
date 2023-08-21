import torch
import fairseq
from fairseq.data import Dictionary
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from SpeechLM import SpeechLMConfig, SpeechLM
import soundfile as sf
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

torch.manual_seed(42)
np.random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.INFO)

# Use GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
logger.info(f'Running on : {device}')

# Load pre-trained SpeechLM model
model_path = '/home/polarboar/models/speechlmh_base_checkpoint_clean.pt'
checkpoint = torch.load(model_path)
cfg = SpeechLMConfig(checkpoint['cfg']['model'])
model = SpeechLM(cfg)
model.load_state_dict(checkpoint['model'])
#model.eval()

# Define LabelEncoder
class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )

# Define Lienar Classifier
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):

        features = None
        #with torch.no_grad():
        #    features = self.model.extract_features(input)[0]
        features = self.model.forward(input, features_only=True)['x']
        num_tokens = features.shape[1]
        features = torch.sum(features, dim=1)
        logits = self.classifier(features[0])
        return logits

class GRUSentimenetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.speechlm = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        features = self.speechlm.forward(x, features_only=True)['x']
        _, hidden = self.gru(features)
        output = self.fc(hidden.squeeze(0))
        output = torch.softmax(output, dim=1)
        return output[0]

def load_audio_names(manifest_path, max_keep, min_keep, retry_times=5):
    n_long, n_short = 0, 0
    names, inds, sizes, labels, audios = [], [], [], [], []
    for i in range(retry_times):
        with open(manifest_path) as f:
            root = f.readline().strip()
            for ind, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 3, line
                sz = int(items[1])
                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    names.append(items[0])
                    inds.append(ind)
                    sizes.append(sz)
                    labels.append(items[2])
        if len(names) == 0:
            logger.info(f"Fail to load manifest for the {i} time")
            #time.sleep(1)
            continue
        else:
            break
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )

    return root, names, inds, tot, sizes, labels

def load_audio(root, name):
    wav, _ = sf.read(root + '/' + name)
    return wav

manifest_path = '/home/polarboar/thesis/data_formatted/sa_slue/'
dicts = {}
dicts['classes'] = Dictionary.load('/home/polarboar/thesis/data_formatted/sa_slue/dict.txt')
label_encoder = LabelEncoder(dicts['classes'])
dict_size = len(label_encoder.dictionary.indices)

def create_dataset(inputs, labels, set_size):
    labels_encoded = torch.tensor([label_encoder(label) for label in labels]).long()
    labels_encoded = F.one_hot(labels_encoded, num_classes=dict_size)

    return [[inputs[i], labels_encoded[i]] for i in range(set_size)]

if __name__ == '__main__':


    train_root, train_names, train_inds, train_tot, train_sizes, train_labels = load_audio_names(manifest_path+'train.tsv', min_keep=None, max_keep=None)
    valid_root, valid_names, valid_inds, valid_tot, valid_sizes, valid_labels = load_audio_names(manifest_path+'valid.tsv', min_keep=None, max_keep=None)
    
    # Create Classifier
    input_dim = 768
    output_dim = dict_size

    #classifier = SentimentClassifier(input_dim, output_dim)
    classifier = GRUSentimenetClassifier(input_dim, input_dim, output_dim)

    # Load Data
    train_size = len(train_labels)
    valid_size = len(valid_labels)

    logger.info(f'Train Size: {train_size}, Validation Size: {valid_size}')
    train_dataset = create_dataset(train_names, train_labels, train_size)
    valid_dataset = create_dataset(valid_names, valid_labels, valid_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    # Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.0001)
    optimizer_model = optim.Adam(model.parameters(), lr=0.0001)

    classifier.to(device)

    # Train Classifier
    num_epochs = 10

    log_interval = 1000

    best_train_accuracy, best_valid_accuracy = 0, 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        classifier.train()
        model.train()
        for idx, (inputs, labels) in zip(range(train_size), train_loader):
            inputs = torch.tensor(np.array([load_audio(train_root, inputs[0])])).to(torch.float)
            inputs = inputs.to(device)
            optimizer_classifier.zero_grad()
            optimizer_model.zero_grad()
            outputs = classifier(inputs)
            labels = labels.flatten().to(torch.float)
            labels = labels.to(device)
            loss = criterion(outputs, labels)

            # Collect statistics
            train_loss += loss
            #predicted = torch.argmax(outputs)
            #label_idx = torch.argmax(labels)
            total += 1
            correct += 1 if torch.argmax(outputs) == torch.argmax(labels) else 0

            # Propagate Loss and update weights
            loss.backward()
            optimizer_classifier.step()
            optimizer_model.step()

            if (idx+1)%log_interval == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Trained: {idx+1}/{train_size}')

        #logger.info()
        logger.info(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss/len(train_loader):.4f}, '
            f'Train Accruacy: {correct*100/total:.2f}')
        best_train_accuracy = max(best_train_accuracy, correct*100/total)
        
        # Validate
        logger.info(f'Evaluating model on validation set....')
        classifier.eval()
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = torch.tensor(np.array([load_audio(valid_root, inputs[0])])).to(torch.float)
                inputs = inputs.to(device)
                outputs = classifier(inputs)
                labels = labels.flatten().to(torch.float)
                labels = labels.to(device)
                valid_loss += criterion(outputs, labels)
                _, predicted = torch.max(outputs, 0)
                _, label_idx = torch.max(labels, 0)
                total += 1
                correct += (predicted == label_idx).sum().item()

        logger.info(f'Valid Loss: {valid_loss/len(valid_loader):.4f}, '
            f'Valid Accruacy: {correct*100/total:.2f}')
        best_valid_accuracy = max(best_valid_accuracy, correct*100/total)

        logger.info(f'Best Train Accuracy: {best_train_accuracy:.2f}, Best Valid Accuracy: {best_valid_accuracy:.2f}')
        

        

import torch
from transformers import SpeechT5Model, SpeechT5Config

# Load the model configuration
config = SpeechT5Config.from_pretrained("microsoft/speecht5_asr")

# Create an instance of the SpeechT5Model
model = SpeechT5Model(config)
model.load_state_dict(torch.load("/disk/scratch1/ramons/data/t2s-xling/speecht5/speecht5_large.pt"))


# Create an instance of the SpeechT5Model
#model = SpeechT5Model(config)

# Load the saved model state dictionary

# Load the state dictionary into the model
#model.load_state_dict(state_dict)















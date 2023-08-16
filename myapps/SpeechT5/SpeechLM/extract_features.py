import torch
import torch.nn.functional as F
from SpeechLM import SpeechLMConfig, SpeechLM
import numpy as np

checkpoint = torch.load('/home/polarboar/models/speechlmh_base_checkpoint_clean.pt')
cfg = SpeechLMConfig(checkpoint['cfg']['model'])
model = SpeechLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

wav_input_16khz = torch.randn(2,20000)
print(f'wav_input: {wav_input_16khz.shape}')
normalize = checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
print(f'normalize: {normalize}')
if normalize:
    wav_input_16khz = F.layer_norm(wav_input_16khz[0], wav_input_16khz[0].shape).unsqueeze(0)

# extract the representation of last layer
rep = model.extract_features(wav_input_16khz)[0]

print(f'rep shape: {rep.shape}')

# extract the representation of each layer
output_layer = model.cfg.encoder_layers + model.cfg.text_transformer.encoder.layers
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=True)[0]
layer_reps = np.array([x.transpose(0, 1) for x in layer_results])

print(f'rep shape: {rep.shape}')
print(f'layer_reps shape: {layer_reps.shape}')
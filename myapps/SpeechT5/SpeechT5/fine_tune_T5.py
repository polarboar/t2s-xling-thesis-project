import torch
from speecht5.tasks.speecht5 import SpeechT5Task
from speecht5.models.speecht5 import T5TransformerModel

checkpoint = torch.load('/path/to/speecht5_checkpoint')

checkpoint['cfg']['task'].t5_task = 'pretrain'
checkpoint['cfg']['task'].hubert_label_dir = "/path/to/hubert_label"
checkpoint['cfg']['task'].data = "/path/to/tsv_file"

task = SpeechT5Task.setup_task(checkpoint['cfg']['task'])
model = T5TransformerModel.build_model(checkpoint['cfg']['model'], task)
model.load_state_dict(checkpoint['model'])

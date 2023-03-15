# SpeechT5

## Important Note
- I substituted the original SpeechT5 README for ours. You can check the origianl one [here](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5)

## Scripts
- `activate.sh` this script is in `.gitignore` because it is machine dependent.
- `create_sa_slue_data.sh` preapre data manifest to fine-tune on SLUE's Sentimenat Analysis (SA) task.
- `fine_tune_sa_slue.sh` fine-tune an SA SLUE set.
- `decode_sa_slue.sh` decode an SA SLUE set and compute metrics: Accuracy, Precission, Recall, and F1. Because it is a multi class classification task, we provide the macro-averaged version -- consistent with [previous work](https://arxiv.org/pdf/2111.10367.pdf).

## Dependencies
- SpeechT5 can be used with FAIRseq, and HuggingFace.

- For now we use the FariSeq version:
    - Install fairseq  (follow instruction in [SpeechT5 github](https://github.com/microsoft/SpeechT5/tree/main/SpeechT5))
    - Install espnet: `pip install espnet`
    - Install tensorboard `pip install tensorboard`
    
- [NOT WORKING YET] To use Hugging Face model, install HF as:
    - Install as `pip install git+https://github.com/huggingface/transformers.git`
    - Install torch audio `pip install torchaudio`
    - Install espnet `pip install espnet`
    - [Here](https://huggingface.co/mechanicalsea/speecht5-sid/tree/main) you can find the structure of the manifest for fairseq

## TODO
- Adapt a task `SpeechT5/speecht5/task` to go from `text->class`
    - Concretely, now I am working creating a data loader in `SpeechT5/speecht5/data`
- Decode a speech (fine-tuned) model with text
- Decode a text  (fine-tuned) model with speech


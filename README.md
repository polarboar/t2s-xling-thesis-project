# t2s-xling
Knowledge transfer between languages based on different modalities (speech and text)

## Models

- [SpeechT5](https://github.com/microsoft/SpeechT5):
    - Multimodal (speech + text) pre-trained model.
    - Follow the instructions [here](https://github.com/EdinburghNLP/t2s-xling/tree/main/myapps/SpeechT5/SpeechT5)
    - [Here](https://github.com/EdinburghNLP/t2s-xling/tree/main/myapps/SpeechT5/SpeechT5) you will a TODO list of things that we are currently implementing

- [SLUE Baselines](https://github.com/sshon-asapp/slue-toolkit):
    - Here we have speech only models (only one modality)
    - For now, we are replicating [ft-w2v2-base-senti.sh](ft-w2v2-base-senti.sh).
    - We plan to port their evaluation scripts to SpeechT5.


## Test sets

- [GLUE](https://github.com/nyu-mll/jiant):
    - text datasets: Glue, Super Glue, and XTREME datasets
    - [here all supported tasks](https://github.com/CompVis/latent-diffusion/issues/207)
    - you require python 3.8 (not higher, no lower)
    - `pip install jiant`
    - `pip install packaging==21.3` [here](https://github.com/CompVis/latent-diffusion/issues/207)
    - `cd ./data/prepare_data; python prepare_text_sets.py`
    - **Dataset Prepared in Hugging Face:** If we decide to go for Hugging Face datasets are usually preapred there
        - SST-2


- [SLUE](https://github.com/asappresearch/slue-toolkit):
    - speech datasets: sentiment analysis for now
    - `pip install git+https://github.com/asappresearch/slue-toolkit.git`


## Important Notes

- In experiment folders (e.g., ./myaps/hf) you need to add your own `./activate.sh`. this script should activate your environment.

import torch
from transformers import SpeechT5Model, SpeechT5Config


# Load HF model
hf_config = SpeechT5Config()
hf_model = SpeechT5Model(hf_config)
variable_names = hf_model.state_dict().keys()

# Load PT model
pt_model = torch.load("/disk/scratch1/ramons/data/t2s-xling/speecht5/speecht5_base.pt")

for key_pt in list(pt_model["model"].keys()):
    if("encoder.version" != key_pt):
            
            pt_model["model"][key_pt.replace("encoder.","encoder.wrapped_encoder.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("encoder.","encoder.wrapped_encoder.")

            pt_model["model"][key_pt.replace("decoder.","decoder.wrapped_decoder.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("decoder.","decoder.wrapped_decoder.")

            
            pt_model["model"][key_pt.replace("self_attn.out_proj.","attention.out_proj.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("self_attn.out_proj.","attention.out_proj.")

            pt_model["model"][key_pt.replace("self_attn.q_proj.","attention.q_proj.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("self_attn.q_proj.","attention.q_proj.")

            pt_model["model"][key_pt.replace("self_attn.v_proj.","attention.v_proj.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("self_attn.v_proj.","attention.v_proj.")

            pt_model["model"][key_pt.replace("self_attn.k_proj.","attention.k_proj.")] = pt_model["model"].pop(key_pt)
            key_pt = key_pt.replace("self_attn.k_proj.","attention.k_proj.")


hf_model.load_state_dict(pt_model["model"])






















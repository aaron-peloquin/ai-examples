# Setup:
# git clone this repo as a sibling of this repo: https://github.com/oobabooga/GPTQ-for-LLaMa/
# in GPTQ-for-LLaMa, runs `python setup_cuda.py install`

# Source: https://github.com/oobabooga/text-generation-webui/blob/main/modules/GPTQ_loader.py
import inspect
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

import modules.shared as shared

sys.path.insert(0, str(Path("..\GPTQ-for-LLaMa")))

try:
    from modelutils import find_layers
except ImportError:
    from utils import find_layers

try:
    from quant import make_quant
    is_triton = False
except ImportError:
    import quant
    is_triton = True

# This function is a replacement for the load_quant function in the
# GPTQ-for_LLaMa repository. It supports more models and branches.
def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=None, kernel_switch_threshold=128, eval=True):
    eval=True
    exclude_layers = exclude_layers or ['lm_head']

    def noop(*args, **kwargs):
        pass


    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()

    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    if not is_triton:
        gptq_args = inspect.getfullargspec(make_quant).args

        make_quant_kwargs = {
            'module': model,
            'names': layers,
            'bits': wbits,
        }
        if 'groupsize' in gptq_args:
            make_quant_kwargs['groupsize'] = groupsize
        if 'faster' in gptq_args:
            make_quant_kwargs['faster'] = faster_kernel
        if 'kernel_switch_threshold' in gptq_args:
            make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

        make_quant(**make_quant_kwargs)
    else:
        quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    model.seqlen = 2048
    return model

# Custom code
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline

base_model = _load_quant(
    model="..\\ai-models\\vicuna-13b-GPTQ-4bit-128g",
    checkpoint="..\\ai-models\\vicuna-13b-GPTQ-4bit-128g\\vicuna-13b-4bit-128g.safetensors",
    wbits=4,
    groupsize=128,
    faster_kernel=False,
    exclude_layers=None,
    kernel_switch_threshold=128,
    )

tokenizer = AutoTokenizer.from_pretrained(
    Path("..\\ai-models\\vicuna-13b-GPTQ-4bit-128g"),
    trust_remote_code=True,
)

pipe = pipeline(
    "text-generation",
    model=base_model,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    max_length=1152,
    temperature=0.34,
    top_p=0.95,
    repetition_penalty=1.2,
)

llm = HuggingFacePipeline(pipeline=pipe)

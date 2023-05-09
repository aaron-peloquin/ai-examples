from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# Note: Not working

model_path = ".\models\mpt-7b-instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path,
    trust_remote_code=True,
    local_files_only=True,
    # load_in_8bit=True,
    # device=0,
)
tokenizer = AutoTokenizer.from_pretrained(model_path) 

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

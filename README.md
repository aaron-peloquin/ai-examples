# ai examples

## Setup
### Pip Install
1. clone repo
2. `npm i`
3. `npm run py_install`
3. `npm run setup`
3. `npm run main`

### Initalize DB
Run `python .\setup\initSRD.py`, first time you run this it will auto-download a smaller AI model to your cache, only needs to be ran once

### Setup Model
Pick which `llm` (AI model) to use, change the import in `main.py` to use that model

For Vicuna ([llm_vicuna7b.py](./modules/llm_vicuna7b.py)), download and place the model in `../ai-models/vicuna-7B-1.1-HF`

For PaLM API ([llm_makersuite.py](./modules/llm_makersuite.py)), set your `MAKER_KEY` in `.env`

### Run main script
`python .\main.py`

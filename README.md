# grammar error correction test

### 1. Gector

`
./predict
`

```shell
python predict.py --model_path ./model/bert_0_gectorv2.th \
                  --vocab_path ./data/output_vocabulary \
		  --input_file ./data/input.txt \
                  --output_file ./output/bert_output.txt \
		  --special_tokens_fix 0 \
		  --transformer_model bert
```



### 2. Gramformer

### 3. salesken - grammar_correction

```python
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("salesken/grammar_correction")  
model = AutoModelForCausalLM.from_pretrained("salesken/grammar_correction").to(device)

input_query="what be the reason for everyone leave the company"
query= "<|startoftext|> " + input_query + " ~~~"


input_ids = tokenizer.encode(query.lower(), return_tensors='pt').to(device)
sample_outputs = model.generate(input_ids,
                                do_sample=True,
                                num_beams=1, 
                                max_length=128,
                                temperature=0.9,
                                top_p= 0.7,
                                top_k = 5,
                                num_return_sequences=3)
corrected_sentences = []
for i in range(len(sample_outputs)):
    r = tokenizer.decode(sample_outputs[i], skip_special_tokens=True).split('||')[0]
    r = r.split('~~~')[1]
    if r not in corrected_sentences:
        corrected_sentences.append(r)

print(corrected_sentences)
```

### reference
- https://github.com/grammarly/gector
- https://github.com/PrithivirajDamodaran/Gramformer
- https://huggingface.co/salesken/grammar_correction

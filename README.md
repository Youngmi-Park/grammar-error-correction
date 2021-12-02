# grammar error correction test

### 1. GECToR

#### download pretrained models
- BERT <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th">[link]</a>
- RoBERTa <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th">[link]</a>
- XLNet <a href="https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th">[link]</a>

#### prediction using pretrained models
`
cd gector/predict
sh predict_bert.sh
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

#### installation
```shell
pip3 install -U git+https://github.com/PrithivirajDamodaran/Gramformer.git
```

#### Corrector
```python
from gramformer import Gramformer
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(1212)

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

influent_sentences = [
    "He are moving here.",
    "I am doing fine. How is you?",
    "How is they?",
    "Matt like fish",
    "the collection of letters was original used by the ancient Romans",
    "We enjoys horror movies",
    "Anna and Mike is going skiing",
    "I walk to the store and I bought milk",
    " We all eat the fish and then made dessert",
    "I will eat fish for dinner and drink milk",
    "what be the reason for everyone leave the company",
]   

for influent_sentence in influent_sentences:
    corrected_sentences = gf.correct(influent_sentence, max_candidates=1)
    print("[Input] ", influent_sentence)
    for corrected_sentence in corrected_sentences:
      print("[Correction] ",corrected_sentence)
    print("-" *100)
```

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
- [GECToR – Grammatical Error Correction: Tag, Not Rewrite Paper](https://aclanthology.org/2020.bea-1.16/)
- [GECToR – Grammatical Error Correction: Tag, Not Rewrite GitHub](https://github.com/grammarly/gector)
- [Gramformer GitHub](https://github.com/PrithivirajDamodaran/Gramformer)
- [salesken Hugging Face](https://huggingface.co/salesken/grammar_correction)

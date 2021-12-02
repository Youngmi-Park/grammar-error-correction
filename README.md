# grammar-error-correction
grammar error correction

Gector

./predict

python predict.py --model_path ./model/bert_0_gectorv2.th \
                  --vocab_path ./data/output_vocabulary \
		  --input_file ./data/input.txt \
                  --output_file ./output/bert_output.txt \
		  --special_tokens_fix 0 \
		  --transformer_model bert




Gramformer

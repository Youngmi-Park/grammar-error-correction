python predict.py --model_path ../model/roberta_1_gectorv2.th \
                  --vocab_path ../data/output_vocabulary \
		  --input_file ../data/input.txt \
                  --output_file ../output/roberta_output.txt \
		  --special_tokens_fix 1 \
		  --transformer_model roberta


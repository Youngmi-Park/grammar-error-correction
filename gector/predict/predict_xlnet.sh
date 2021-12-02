python predict.py --model_path ../model/xlnet_0_gectorv2.th \
                  --vocab_path ../data/output_vocabulary \
		  --input_file ../data/input.txt \
                  --output_file ../output/xlnet_output.txt \
		  --special_tokens_fix 0 \
		  --transformer_model xlnet


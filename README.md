# NLP_Sentiment

python run.py --do_train --task nli --dataset snli --output_dir ./trained_model/

python run.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./eval_output/


python code/run.py --do_train --task nli --dataset snli --output_dir ./code/trained_model/ --max_train_samples 100 --max_eval_samples 20


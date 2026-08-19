# for code folder 
python app/main.py --do_train --task nli --dataset snli --output_dir ./code/trained_model/ --max_train_samples 100 --max_eval_samples 20

to force reload datasets: 
python app/main.py --do_train --task nli --dataset snli --output_dir ./code/trained_model/ --load_dataset


# for app folder, more functionality 
### base command 
python app/main.py  --output_dir ./code/trained_model/ 

### make challenge set for example 
python app/main.py  --output_dir ./code/trained_model/  --make_datasets 'challenge'

### train on challenge set for example 
python app/main.py --do_train --task nli --dataset './datasets/challenge_set.hf' --output_dir ./code/trained_model/ 


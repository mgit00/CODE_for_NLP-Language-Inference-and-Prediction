from pathlib import Path
import logging 
import datasets 
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, compute_accuracy
import os
import json
import torch 

from make_datasets import MakeDatasets

class Driver: 
    '''
    this is the driver class 
    the filenames will be stagnant for now 
    later implementations may give more flexible filenameing
    '''

    def __init__(self, training_args, args): 
        self.logger = logging.getLogger('main_logger')

        self.training_args = training_args
        self.args= args 

        if args.make_datasets: 
            # if the make datasets is called, then this will be the path of the pipeline
            # call making datasets class 
            self.logger.debug('make datasets')
            MakeDatasets(args)  

        else: 
            self.logger.debug('do model train/eval, not making any datasets')
            # continue with model creation 
            self.model_train_eval()

             

    
    def model_train_eval(self) -> None : 
        '''
        this gets the model and tokenizer form huggingface 
        args.model default is ELECTRA, can also use BERT
            - 'google/electra-small-discriminator'
            - 'bert-base-uncased'
        '''

        NUM_PREPROCESSING_WORKERS= 2 

        training_args= self.training_args 
        args= self.args

        dataset_id= ('snli',)
        eval_split = 'validation'
        dataset= load_from_disk(args.dataset)


        model = AutoModelForSequenceClassification.from_pretrained(self.args.model, {'num_labels': 3}  )
        tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)

        # Select the dataset preprocessing function (these functions are defined in helpers.py)
        if args.task == 'nli':
            # create functions: prepare_train_dataset, prepare_eval_dataset
                # functions used integrate with the Hugging Face datasets.map()
                # prepare_eval_dataset = prepare_dataset_nli
                # exs mean raw data examples 
            prepare_train_dataset = prepare_eval_dataset = lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
            
        else:
            raise ValueError('have not built out task yet: {}'.format(args.task))

        print("Preprocessing data... -this takes a little bit, should only happen once per dataset")
        if dataset_id == ('snli',):
            # remove SNLI examples with no label
            dataset = dataset.filter(lambda ex: ex['label'] != -1)

        self.logger.info('dataset filtered')

        # choose train or eval based on modes chosen, can do both 
        train_dataset = None # raw exs from dataset['train']
        eval_dataset = None
        train_dataset_featurized = None
        eval_dataset_featurized = None
        if training_args.do_train:
            train_dataset = dataset['train']
            if args.max_train_samples: # if specified, limit the number of examples to include 
                train_dataset = train_dataset.select(range(args.max_train_samples))
            train_dataset_featurized = train_dataset.map( # useable with HF Dataset objct 
                prepare_train_dataset, # function defined above 
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=train_dataset.column_names # drops unneeded raw data columns immediately after processing text
            )
        if training_args.do_eval:
            eval_dataset = dataset[eval_split]
            if args.max_eval_samples:
                eval_dataset = eval_dataset.select(range(args.max_eval_samples))
            eval_dataset_featurized = eval_dataset.map(prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=eval_dataset.column_names
            )
            
        # ------- 
        # Select the training configuration
            # use HF Trainer 
                # Manages epochs, batches, forward passes, backpropagation, and weight updates 
                # Periodically runs your evaluation dataset and calculates metrics (like Accuracy or F1-score) during training
                # Periodically saves the model weights and training state so you can resume training if it crashes
                # Automatically detects and moves model and data to CPU or GPU
        trainer_class = Trainer # HF Trainer
        print('torch', torch.cuda.is_available())  # Must return True
        print(torch.cuda.get_device_name(0))
    
        eval_kwargs = {}
        
        compute_metrics = compute_accuracy # function for nli 
        
        # This function wraps the compute_metrics function, storing the model's predictions
        # so that they can be dumped along with the computed metrics
        eval_predictions = None # EvalPrediction object 
        def compute_metrics_and_store_predictions(eval_preds): # fed to HF Trainer, eval_preds = EvalPrediction object 
            nonlocal eval_predictions
            eval_predictions = eval_preds
            return compute_metrics(eval_preds)
    
        # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
        trainer = trainer_class(
            model=model, # custom arg, downloaded electra or bert 
            args=training_args, # HFArgumentParser defaults 
            train_dataset=train_dataset_featurized, # processed train data 
            eval_dataset=eval_dataset_featurized, 
            processing_class=tokenizer, # old transformers package: tokenizer= AutoTokenizer. from_pretrained electra or bert model 
            compute_metrics=compute_metrics_and_store_predictions # also stores the EvalPrediction object, not letting it fall out of memory  
        )
    
        print('HF Trainer initialized')
    
        # Train and/or evaluate process 
        if training_args.do_train:
            trainer.train()
            trainer.save_model() # to HFArgumentParser default location 
            # can also do tokenizer.save_pretrained()- so I easily reload everything as a pair later
            self.logger.info('training done')
    
        if training_args.do_eval:
            results = trainer.evaluate(**eval_kwargs) #### 
            print('Evaluation results:')
            print(results)
    
            # make directory from namespace args 
            os.makedirs(training_args.output_dir, exist_ok=True)
    
            # # evalutaion metrics 
            # with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            #     json.dump(results, f)
    
            # evaluation predictions 
            with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
                for i, example in enumerate(eval_dataset): # raw eval exs example data 
                    example_with_prediction = dict(example) # 
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
    
            self.logger.info('eval done, saved as well')

        return None
    
import datasets
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, compute_accuracy
import os
import json
import torch 

NUM_PREPROCESSING_WORKERS = 2

'''
task = nli 
dataset  = snli
dataset_id= 'snli'

HF DatasetDict and Dataset won't overwhelm memory, used Apache Arrow memory-mapping
    # use .map() to take advantage of tokenizing from this type of data storage 
    - for Dataset: 
        - can access 'premise' column (as list) like a py dict: test['premise']
        - access row like a list:  test[0]
        - can slice like np array : test[0:5]


DatasetDict({
    test: Dataset({
        features: ['premise', 'hypothesis', 'label'], # 
        num_rows: 10000
    })
    validation: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 10000
    })
    train: Dataset({
        features: ['premise', 'hypothesis', 'label'],
        num_rows: 550152
    })
})
'''

def main():
    # HfArgumentParser gives default values 
    # to be specified when running: 
        # --do_train
        # --do_eval
        # --per_device_train_batch_size <int, default=8> training batch size
        # --num_train_epochs <float, default=3.0>
        # --output_dir <path> for model checkpoints, required 

    argp = HfArgumentParser(TrainingArguments) 
  
    # This argument specifies the base model to fine-tune.
    #     This should either be a HuggingFace model ID (see https://huggingface.co/models)
    #     or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin)
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator')
    # This argument specifies which task to train/evaluate on.
    #     Pass "nli" for natural language inference or "qa" for question answering.
    #     By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True)
    # This argument overrides the default dataset used for the specified task.
    argp.add_argument('--dataset', type=str, default=None)
    # This argument limits the maximum sequence length used during training/evaluation.
    #     Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated
    argp.add_argument('--max_length', type=int, default=128)
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--load_dataset', action='store_true', default=None)


    training_args, args = argp.parse_args_into_dataclasses() # default training args from HF, namespace args 

    

    # Dataset selection
    # For SNLI, you can prepare a file with each line containing one
        # example as follows:
        # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    
    default_datasets = {'nli': ('snli',)} # NLI 
    dataset_id = default_datasets[args.task]
    # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
    eval_split = 'validation'

    if args.load_dataset: 
      dataset= load_from_disk('./code/datasets/snli_dataset')

    else: 
      # Load the raw data
          # *dataset_id unpacks elements by position
          # **dataset_id_dictionary unpacks elements by name 
      dataset = datasets.load_dataset('stanfordnlp/snli') #  # call for HF dataset: dataset_id= 'snli'. * to unpack tuple into positional func args 
      
      dataset.save_to_disk('./code/datasets/snli_dataset')
      print('loaded in dataset', dataset_id)
 

    # NLI models need to have the output label count specified 
        # (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} 
    # Here we select the right model fine-tuning head
    model_classes = {'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task] # 'nli'
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
        # AutoModelForSequenceClassification .from_pretrained('google/electra-small-discriminator' , **{'num_labels': 3}  ) 
        # can also use bert : "bert-base-uncased"
    model = model_class.from_pretrained(args.model, **task_kwargs) # unpack ** dictionary args into name-based values
    # tokenizer-- match to model 
        # AutoTokenizer .from_pretrained('google/electra-small-discriminator'  ) 
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

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

    print('dataset filtered')

    if args.load_dataset: 
      train_dataset = dataset['train']
      eval_dataset = dataset[eval_split]
      # dataset= dataset # load_from_disk('./code/datasets/snli_dataset')
      train_dataset_featurized = load_from_disk('./code/datasets/snli_train_featurized')
      eval_dataset_featurized = load_from_disk('./code/datasets/snli_eval_featurized')

    else: 
      # choose train or eval based on modes chosen, can do both 
      train_dataset = None # raw exs from dataset['train']
      eval_dataset = None
      train_dataset_featurized = None
      eval_dataset_featurized = None
      # if training_args.do_train:
      if True: 
          train_dataset = dataset['train']
          if args.max_train_samples: # if specified, limit the number of examples to include 
              train_dataset = train_dataset.select(range(args.max_train_samples))
          train_dataset_featurized = train_dataset.map( # useable with HF Dataset objct 
              prepare_train_dataset, # function defined above 
              batched=True,
              num_proc=NUM_PREPROCESSING_WORKERS,
              remove_columns=train_dataset.column_names # drops unneeded raw data columns immediately after processing text
          )
      # if training_args.do_eval:
      if True: 
          eval_dataset = dataset[eval_split]
          if args.max_eval_samples:
              eval_dataset = eval_dataset.select(range(args.max_eval_samples))
          eval_dataset_featurized = eval_dataset.map(prepare_eval_dataset,
              batched=True,
              num_proc=NUM_PREPROCESSING_WORKERS,
              remove_columns=eval_dataset.column_names
          )

      train_dataset_featurized.save_to_disk('./code/datasets/snli_train_featurized')
      eval_dataset_featurized.save_to_disk('./code/datasets/snli_eval_featurized')


    # train_dataset, eval_dataset = raw data 
    # train_dataset_featurized, eval_dataset_featurized = tokenized data (AutoTokenizer + .map() for HF Dataset)
    print('datasets put through autotokenizer')

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
        print('training done')

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
        print('here')
        print(eval_predictions)
        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            for i, example in enumerate(eval_dataset): # raw eval exs example data 
                example_with_prediction = dict(example) # 
                example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                f.write(json.dumps(example_with_prediction))
                f.write('\n')

        print('eval done, saved as well')

if __name__ == "__main__":
    main()

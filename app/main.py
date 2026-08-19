from transformers import HfArgumentParser, TrainingArguments

from driver import Driver 
import logging

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
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], default= 'nli', required=False)
    # This argument overrides the default dataset used for the specified task.
    argp.add_argument('--dataset', type=str, default=None)
    # This argument limits the maximum sequence length used during training/evaluation.
    #     Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated
    argp.add_argument('--max_length', type=int, default=128)
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--make_datasets', nargs ='?', 
                      choices= ['snli', 'challenge', 'contrast', 'text_attack', 'hardest_10', 'heuristics_only' ], 
                      help= 'force reload datasets')


    training_args, args = argp.parse_args_into_dataclasses() # default training args from HF, namespace args 

    return [training_args, args] 


def make_logger(): 
    # Create the logger
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging.DEBUG)

    # Use StreamHandler 
    console_handler = logging.StreamHandler()

    # Create and set a format
    log_format = logging.Formatter(' %(levelname)s | %(message)s')
    console_handler.setFormatter(log_format)

    # Add handler to logger
    logger.addHandler(console_handler)

    logger.info('logger created ')
    return logger 

if __name__ == "__main__":
    args_tuple= main()
    logger = make_logger()
    Driver(*args_tuple) # positional args 


    logger.info('finished')
import logging 
from pathlib import Path

import datasets 
from datasets import DatasetDict, load_from_disk 

from making_datasets.create_challenge_set import Challenge_Set
from making_datasets.create_contrast_set import Contrast_Set
from making_datasets.create_hardest_examples import Hardest_Examples_Set
from making_datasets.create_hypothesis_set import hypothesis_testing_data_alteration

class MakeDatasets: 
    '''
    this class makes the datasets
    the datasets will be saved to the directory 
        folder is: app/datasets

    choices= ['snli', 'challenge', 'contrast', 'text_attack', 'hardest_10', 'heuristics_only' ]
    '''

    def __init__(self, args): 
        self.logger = logging.getLogger('main_logger')

        # create datafolder if needed 
        folder_path = Path("./app/datasets") # ./ means from the call location of root 
        folder_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug('dataset folder exists now')

        self.args= args 
        dataset_type = self.args.make_datasets

        self.logger.info('making the dataset: ' + self.args.make_datasets)
        if dataset_type == 'snli': 
            self.make_snli()
        elif dataset_type == 'challenge': 
            self.make_challenge()
            pass
        elif dataset_type == 'contrast': 
            self.make_contrast()
            pass 
        elif dataset_type == 'text_attack': 
            # uses create_text_attack_set.py 
            # data made in ipynb file
            pass 
        elif dataset_type == 'hardest_10':
            self.make_hardest_10() 
            pass 
        elif dataset_type =='heuristics_only': 
            self.make_heuristics_only()
            pass 
    
    def make_snli(self) -> None: 
        '''
        this class loads the snli dataset
        this only has to be called once
        '''
        # snli dataset has to be loaded 
            # load the smaller dataset for now, change for gpu usage 
                
        snli_path = './app/making_datasets/snli'
        limits = {'train': 1000, 'validation': 100, 'test': 100}


        # filepath = Path(snli_path)
        # if filepath.is_file() and self.args.load_dataset==False:
        #     self.logger.debug('snli dataset is already loaded') 
        #     return 

        
        # Load the raw data
        dataset = datasets.load_dataset('stanfordnlp/snli')
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

                
        limited_dataset= {}
        for split, data in dataset.items(): 
            limited_dataset[split] = data.select(range(limits[split])) 
        limited_dataset= DatasetDict(limited_dataset)
        self.logger.debug('limited dataset is made' + str(limited_dataset))

        # save the limited dataset now 
        limited_dataset.save_to_disk(snli_path)
        self.logger.info('limited dataset now saved to:' + snli_path)

        return 

    ## these functions call other classes in the same folder 
    # otherwise, this script gets crowded
    def make_challenge(self) -> None: 
        # place to path = './app/datasets/challenge'
        Challenge_Set()
        return 
    def make_contrast(self) -> None: 
        Contrast_Set()
        return 
    def make_hardest_10(self) -> None: 
        hardest=Hardest_Examples_Set('./app/making_datasets/baseline_eval_predictions_og_val.jsonl')
        return 
    def make_heuristics_only(self) -> None: 
        subsection= load_from_disk('./code/making_datasets/snli')['test']
        hypothesis_testing_data_alteration(subsection, True)
        return 
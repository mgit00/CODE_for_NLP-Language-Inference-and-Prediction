'''
create dataset with adversarial data augmentation through TextAttack 
'''


import datasets 
from datasets import Dataset
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets

import json


class Create_JSONL_File():
    def __init__(self, dataset_obj, filename):
        self.filename=filename
        self.write_jsonl(dataset_obj)

        
    def write_jsonl(self,dataset):
        filename= open(self.filename, encoding='utf-8', mode='w')
        
        for example in dataset: 
            filename.write(json.dumps(example))
            filename.write('\n')

        return None 



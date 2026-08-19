'''
create contrast set
'''

import datasets 
from datasets import Dataset
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets

import csv


class Contrast_Set():
    def __init__(self):
        contrast_filename= './app/making_datasets/inferred_examples.csv'

        dataset= self.get_dataset(contrast_filename)

        self.save_data_obj(dataset, 'contrast_test_set.hf') 
        

    def get_dataset(self, filename):
        change_rule=[] # not used for now 
        premise=[]
        hypothesis=[]
        label=[]

        code_labels=['entailment', 'neutral', 'contradiction']

        text= open('inferred_examples.csv', mode='r')
        text= csv.reader(text)
        
        for example in text:
            if example[3] in code_labels: 
                change_rule.append(example[0])
                premise.append(example[1])
                hypothesis.append(example[2])
                label.append(example[3])
                if example[3] not in ['entailment', 'neutral', 'contradiction']:
                    raise NameError('non standard label')  
            
     
        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])}) 

        form_dataset = Dataset.from_dict({'premise':premise, 'hypothesis':hypothesis, 'label':label}, features=same_features)

        return form_dataset

    
    def save_data_obj(self,dataset, new_txt_file):
        dataset.save_to_disk(new_txt_file)
        return None
    

        
        
            
            

    
        

        

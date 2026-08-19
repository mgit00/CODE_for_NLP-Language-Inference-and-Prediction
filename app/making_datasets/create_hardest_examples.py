'''
create hardest examples dataset
'''

import datasets 
from datasets import Dataset
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets

import jsonlines
import scipy 
from scipy.special import softmax
import numpy as np


# take the Baseline model's failed predictions and find the hardest examples from that
# takes .json file from predictions of baseline model on og validation set

class Hardest_Examples_Set():
    def __init__(self, saved_jsonl_filename, ): # jsonl

        # around 1000 incorrect examples -> 10%
        self.threshold=0.5 # -> 119 examples
        
    
        # non local file because only 1 file is being evauated at a time
        data= self.get_dataset(saved_jsonl_filename)

        self.save_data_obj(data, 'hardest_examples_test_set.hf') 

         
        

    def get_dataset(self,jsonl_name):
        text=jsonlines.open(jsonl_name)

        premise=[]
        hypothesis=[]
        label=[]

        count=0 
        
        for example in text.iter():
            
            if example['label'] == example['predicted_label']: # if they are correct --- maynot have to do this??...
                continue
            
            count=count+1 # if incorrect
            
            if self.check_difficulty(example) == False:
                continue
            
            # now only diffiuclt examples are left
            premise.append(example['premise'])
            hypothesis.append(example['hypothesis'])
            label_=example['label']
            if label_==0:
                label_='entailment'
            elif label_==1:
                label_='neutral'
            elif label_==2:
                label_='contradiction'
            label.append(label_)

        

            
        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})

        form_dataset=Dataset.from_dict({'premise':premise, 'hypothesis':hypothesis, 'label':label}, features=same_features)

        return form_dataset
            

            

    def check_difficulty(self, example):
         difficult= False

         scores= example['predicted_scores']

         soft= softmax(scores)

         if (abs(soft[0]-soft[1])< self.threshold) and (abs(soft[1]-soft[2])< self.threshold):
             difficult=True 

         return difficult




    def save_data_obj(self,dataset, new_txt_file):
        dataset.save_to_disk(new_txt_file)
        return None
        
        
        
            
        
        
        

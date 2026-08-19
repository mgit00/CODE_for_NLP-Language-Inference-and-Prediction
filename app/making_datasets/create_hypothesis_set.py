'''
create hypothesis testing dataset
make premise empty string 
'''

import datasets 
from datasets import Dataset
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets

def hypothesis_testing_data_alteration(dataset_subsection, save_dataset=False): # train, test, validation....
    premise=[]
    hypothesis=[]
    label=[]

    for example in dataset_subsection:
        premise.append('')
        hypothesis.append(example['hypothesis'])
        label.append(example['label'])


    # make dataset
    same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})

    form_dataset = Dataset.from_dict({'premise':premise, 'hypothesis':hypothesis, 'label':label}, features=same_features)
    if save_dataset==True: 
        form_dataset.save_to_disk('formed_hypothesis_testing_dataset.hf')
    
    return form_dataset 
        

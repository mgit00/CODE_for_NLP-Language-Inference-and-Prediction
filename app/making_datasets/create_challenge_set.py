'''
create challenge dataset

--- revised_const_finder redo 
'''

import datasets 
from datasets import Dataset
from datasets import ClassLabel, Features, Value
from datasets import concatenate_datasets

# convert heuristics_train_set.txt to right format
# saves formatted McCoy et al datasets
# adds own data from SNLI to test set (not from og train, otherwise training twice on the same data)
    # use test and validation as train and test 
# gets completed 
class Challenge_Set():
    def __init__(self):
        train='./app/making_datasets/heuristics_train_set.txt'
        test='./app/making_datasets/heuristics_evaluation_set.txt'

    
        # saves formatted McCoy et al datasets
        train_data=self.create_data_obj(train)
        test_data=self.create_data_obj(test)


##        print(train_data.filter(lambda ex: ex['label'] == 0))
##        print(train_data.filter(lambda ex: ex['label'] == 2))
##        
##        print('train_data',train_data)
##        print('test_data',test_data)

        # adds own data from SNLI to test set
        own=Own_Challenge_Data() # instance
        own_train_data=own.train_set
        own_test_data=own.test_set
##        print('own_train_data',own_train_data)
##        print('own_test_data',own_test_data)

        # concatenate paper's data with my own
        completed_train_data=concatenate_datasets([train_data, own_train_data])
        completed_test_data=concatenate_datasets([test_data, own_test_data])


        # create new file
        self.new_train_filename='revised_heuristics_train_set.hf'
        self.new_test_filename='revised_heuristics_test_set.hf'
        
        self.save_data_obj(completed_train_data, self.new_train_filename)
        self.save_data_obj(completed_test_data, self.new_test_filename)
        

    def get_dataset(self, subset_name):
        if subset_name=='train':
            filename=self.new_train_filename
        elif subset_name=='test':
            filename=self.new_test_filename
        else:
            raise NameError('wrong filename')
        
        return datasets.load_from_disk(filename)


    def create_data_obj(self,txt_file):
        text=open(txt_file,'r')

        premise=[]
        hypothesis=[]
        label=[]

        first=True
        for line in text:
            if first==True:
                first=False
                continue
            
            
            line=line.split('\t')
            
            label_=line[0]
            if label_=='non-entailment':
                label_='contradiction'
##            if label_ != "entailment" or label_ !='contradiction':
####                print('here', label_, len(label_)) 
##                continue
                
            
##            print(label_)
            label.append(label_) 
            premise.append(line[5])
            hypothesis.append(line[6])
        
        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})

        form_dataset=Dataset.from_dict({'premise':premise, 'hypothesis':hypothesis, 'label':label}, features=same_features)

        return form_dataset

    
    def save_data_obj(self,dataset, new_txt_file):
        dataset.save_to_disk(new_txt_file)
        return None




import nltk
nltk.download('bllip_wsj_no_aux')
from nltk.data import find
from bllipparser import RerankingParser
model_dir = find('models/bllip_wsj_no_aux').path
parser = RerankingParser.from_unified_model_dir(model_dir)

import re 

class Own_Challenge_Data():
    def __init__(self):
        dataset = datasets.load_dataset('stanfordnlp/snli')

        train_set_lex= self.revised_lex_finder(dataset,'test') # for augmenting the challenge set, don't use already trained on examples with og train set
        train_set_subseq= self.revised_subseq_finder(dataset,'test')
        
        test_set_lex= self.revised_lex_finder(dataset,'validation')
        test_set_subseq= self.revised_subseq_finder(dataset,'validation')

        # concatenate the data fron lex, subseq, and const
##        train_set= concatenate_datasets([train_set_lex, train_set_subseq, train_set_const])
##        test_set= concatenate_datasets([test_set_lex, test_set_subseq, test_set_const])
        self.train_set= concatenate_datasets([train_set_lex, train_set_subseq])
        self.test_set= concatenate_datasets([test_set_lex, test_set_subseq])

        
        


    def revised_lex_finder(self,dataset, subset=None, save_dataset=False): # subset is string: train, test, validation
        if subset==None: # for gathering data statistics 
            data= concatenate_datasets([dataset['train'], dataset['test'],dataset['validation'] ])
        else:
            data=dataset[subset] # shouldn't take from train set as this has already been taken into account by model

        #-
        count_entailment = 0
        count_neutral = 0
        count_contradiction = 0
        #-

        list_premises=[] 
        list_hypotheses=[] 
        list_labels=[] 
        
        for i, example in enumerate(data):
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']

            #--- from lex_finder
            prem_words = []
            hyp_words = []

            for word in premise.split():
                if word not in [".", "?", "!"]:
                    prem_words.append(word.lower())

            for word in hypothesis.split():
                if word not in [".", "?", "!"]:
                    hyp_words.append(word.lower())

            prem_filtered = " ".join(prem_words)
            hyp_filtered = " ".join(hyp_words)

            all_in = True

            for word in hyp_words:
                if word not in prem_words:
                    all_in = False
                    break

            if all_in:
                    
                if label == 0: #"entailment":
                    count_entailment += 1 # not used want to train against heuristic -----... 
    ##                list_premises.append(premise) #  # can't have this at the top b/c there is mismatch of label count from there being a label not in 0-2 
    ##                list_hypotheses.append(hypothesis) # 
    ##                list_labels.append("entailment") #
                if label == 1: #"neutral":
                    count_neutral += 1
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
    ##                print(premise, hypothesis, label)
                    list_labels.append('neutral') # 
                if label == 2: #"contradiction":
                    count_contradiction += 1
    ##                print(premise, hypothesis, label)
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
                    list_labels.append('contradiction') # 

            #print(premise, hypothesis, label)

        print("Entailment:", count_entailment)
        print("Contradiction:", count_contradiction)
        print("Neutral:", count_neutral)


       
        # should have created a list of all the premises hypothesis and labels
            # then, take this to create a dataset object 
        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})
        created_dataset = Dataset.from_dict({'premise':list_premises, 'hypothesis':list_hypotheses, 'label':list_labels}, features=same_features)

        if save_dataset==True: 
            created_dataset.save_to_disk('lex_challenge_set.hf') # derived from og test set 

        return created_dataset

    def revised_subseq_finder(self,dataset, subset=None, save_dataset=False): # subset is string: train, test, validation 
        if subset==None: # for gathering data statistics 
            data= concatenate_datasets([dataset['train'], dataset['test'],dataset['validation'] ])
        else:
            data=dataset[subset] # shouldn't take from train set as this has already been taken into account by model

        #-
        count_entailment = 0
        count_neutral = 0
        count_contradiction = 0
        #-

        list_premises=[] 
        list_hypotheses=[] 
        list_labels=[] 
        
        for i, example in enumerate(data):
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']

            #--- from subseq_finder
            prem_words = []
            hyp_words = []

            for word in premise.split():
                if word not in [".", "?", "!"]:
                    prem_words.append(word.lower())

            for word in hypothesis.split():
                if word not in [".", "?", "!"]:
                    hyp_words.append(word.lower())

            prem_filtered = " ".join(prem_words)
            hyp_filtered = " ".join(hyp_words)

            
            if hyp_filtered in prem_filtered:
                
                if label == 0: #"entailment":
                    count_entailment += 1
    ##                list_premises.append(premise) #  # can't have this at the top b/c there is mismatch of label count from there being a label not in 0-2 
    ##                list_hypotheses.append(hypothesis) # 
    ##                list_labels.append("entailment") #
                if label == 1: #"neutral":
                    count_neutral += 1
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
    ##                print(premise, hypothesis, label)
                    list_labels.append('neutral') # 
                if label == 2: #"contradiction":
                    count_contradiction += 1
    ##                print(premise, hypothesis, label)
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
                    list_labels.append('contradiction') # 

            #print(premise, hypothesis, label)

        print("Entailment:", count_entailment)
        print("Contradiction:", count_contradiction)
        print("Neutral:", count_neutral)


       
        # should have created a list of all the premises hypothesis and labels
            # then, take this to create a dataset object
        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})
        created_dataset = Dataset.from_dict({'premise':list_premises, 'hypothesis':list_hypotheses, 'label':list_labels}, features=same_features)

        if save_dataset==True: 
            created_dataset.save_to_disk('subseq_challenge_set.hf') # derived from og test set 

        return created_dataset


    '''
    uncomment when using the below function

    import nltk
    nltk.download('bllip_wsj_no_aux')
    from nltk.data import find
    from bllipparser import RerankingParser
    model_dir = find('models/bllip_wsj_no_aux').path
    parser = RerankingParser.from_unified_model_dir(model_dir)

    import re 
    ''' 

    def revised_const_finder(self,dataset, subset, save_dataset=False): # subset is string: train, test, validation 
        if subset==None: # for gathering data statistics 
            data= concatenate_datasets([dataset['train'], dataset['test'],dataset['validation'] ])
        else:
            data=dataset[subset] # shouldn't take from train set as this has already been taken into account by model

        #-
        count_entailment = 0
        count_neutral = 0
        count_contradiction = 0
        #-

        list_premises=[] 
        list_hypotheses=[] 
        list_labels=[] 

        first=True # ?? false
        counter=0
        
        for i, example in enumerate(data):
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']
            parsed = parser.simple_parse(premise.lower())
            parse = clean_parse(parsed)
    ##        parse = 0 ### do this... 

            #--- from const_finder
            prem_words = []
            hyp_words = []

            for word in premise.split():
                if word not in [".", "?", "!"]:
                    prem_words.append(word.lower())

            for word in hypothesis.split():
                if word not in [".", "?", "!"]:
                    hyp_words.append(word.lower())

            prem_filtered = " ".join(prem_words)
            hyp_filtered = " ".join(hyp_words)

            
            if hyp_filtered in prem_filtered:
                
                if label == 0: #"entailment":
                    count_entailment += 1
                    list_premises.append(premise) #  # can't have this at the top b/c there is mismatch of label count from there being a label not in 0-2 
                    list_hypotheses.append(hypothesis) # 
                    list_labels.append("entailment") #
                if label == 1: #"neutral":
                    count_neutral += 1
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
    ##                print(premise, hypothesis, label)
                    list_labels.append('neutral') # 
                if label == 2: #"contradiction":
                    count_contradiction += 1
    ##                print(premise, hypothesis, label)
                    list_premises.append(premise) # 
                    list_hypotheses.append(hypothesis) # 
                    list_labels.append('contradiction') # 

            #print(premise, hypothesis, label)

        print("Entailment:", count_entailment)
        print("Contradiction:", count_contradiction)
        print("Neutral:", count_neutral)


       
        # should have created a list of all the premises hypothesis and labels
            # then, take this to create a dataset object

        same_features=Features({'premise': Value(dtype='string', id=None), 'hypothesis': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])})
        created_dataset = Dataset.from_dict({'premise':list_premises, 'hypothesis':list_hypotheses, 'label':list_labels}, features=same_features)

    ##    data=concatenate_datasets([ad_chall_test_set, contrast_set])
        if save_dataset==True: 
            created_dataset.save_to_disk('const_challenge_set.hf') # derived from og test set 

        return created_dataset


    def clean_parse(parse):
    ##    print(parse)
    ##    print(type(parse))
        parse=str(parse)

        parse =re.sub('S1', r'',parse)
        parse =re.sub(':', r'',parse)
        parse =re.sub('\)', r' )',parse)
        parse =re.sub('\.\ \.', r' .',parse)
        parse =re.sub('([A-Z])', r'',parse)

        return parse
        
        
        
        

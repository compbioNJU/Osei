import torch
import torch.nn as nn

class NucPreprocess():
    def __init__(self, sequences):
        self.sequences = sequences
        self.classes = 5 #A,T,C,G,N
        self.encoding_dict = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4, 'Y':4, 'R':4, 'K':4, 'M':4, 'S':4, 'W':4, 'V':4, 'B':4, 'H':4, 'D':4}
        self.decoding_dict = {0:'A', 1:'T', 2:'C', 3:'G', 4:'N'}
        
        
    def onehot_for_nuc(self):
        x = list()
        for seq in self.sequences:
            precoding = [ self.encoding_dict[nuc.upper()] for nuc in seq ]
            classes = self.classes
            index = torch.unsqueeze(torch.tensor(precoding).long(),dim=1)
            src = torch.ones(len(seq), classes).long()
            
            onehot = torch.zeros(len(seq), classes).long()
            onehot.scatter_(dim=1, index=index, src=src)
            
            # Droup N base dim in coding
            onehot = onehot[:,:4]
            x.append(onehot.short())
            
        return x
        
        
    def decode_for_nuc(self, coded_seq):
        source_seq = ''
        for coding in coded_seq:
            if torch.sum(coding):
                new_nuc = self.decoding_dict[int(coding.nonzero().squeeze())]
                source_seq += new_nuc
            else:
                source_seq += 'N'
                
        return source_seq
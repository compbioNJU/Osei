import torch
from plotnine import *
import pandas as pd
from Bio import SeqIO
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


def load_data(path, ftype='fasta', label=False, strand=False, pos=False):
    '''
        Loading data from specific file path. Default file type is fasta.
        The function returns two list: seq_list and seq_label
        If label equals to False, returns seq_list
        If strand equals to True, return complement sequnece as well
        If pos equals to True, return sequences position info as well
     '''
    seqs = []
    if label:
        seqlabels = []
    if pos:
        seqpos = []

    for seq in SeqIO.parse(path, ftype):
        #if 'N' not in seq.seq and len(seq) == 300:
        seqs.append(str(seq.seq))
        if strand:
            seqs.append(str(seq.seq.complement()))
        
        seqName = seq.id
        if label:
            seqlabels.append(seqName.split('::')[0])
            if strand:
                seqlabels.append(seqName.split('::')[0])
                
        if pos:
            seqpos.append(seqName.split('::')[1])
            if strand:
                seqpos.append(seqName.split('::')[1])

    if label and pos:
        return seqs, seqlabels, seqpos
    elif label:
        return seqs, seqlabels
    elif pos:
        return seqs, seqpos
    else:
        return seqs


def draw_scores_distribution(scores_list):
    '''
        Produce scores(labels)'s values distribution
        Annotate metrics of distribution too
    '''
    x = list(range(len(scores_list)))
    y = scores_list
    df = pd.DataFrame({'x':x, 'y':y})
    
    p = (ggplot(df, aes(x='x', y='y'))+geom_point(size=.1, color='#9ecae1')+geom_line(size=.05, color='#9ecae133')+theme_bw()+
     annotate(geom='text',x=max(x)-500,y=max(y)-500,label='Scores\nNum:{}\nMin:{:.2f}\nMean:{:.2f}\nMax{:.2f}'.format(len(y),min(y),np.mean(y),max(y)), 
              color='#fdae6b', size=8))
    
    return p


def tag_encode(tags, tag_dict, sep=','):
    '''
        Encode tags as binary vectors representing the absence/presence of the Profiles
    '''
    result = []
    for tag in tags:
        tmp_result = [0] * len(tag_dict)
        tag = tag.split(sep)
        for item in tag:
            tmp_result[tag_dict[item]] = 1
        result.append(tmp_result)
        
    return np.array(result)


def evaluation(outputs, labels):
    '''
        Function that can evaluate models prediction classes with real data label
        Return correct rate
    '''
    # outputs => probability (float)
    # labels => labels
    # 根据outputs的概率分配label，跟真实的label做比较，输出正确率
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    
    return correct


def draw_metrics(x, y, anno, model=None, metric='ROC'):
    '''
        Function that can produce ROC curve using (fpr, tpr) and PR curve using (recall, precision)
        Parameters `metric` should only be ROC or PR, anno could be arbitrary
    '''
    if metric == 'ROC':
        fpr = x
        tpr = y
        au_roc = anno
        if model is not None:
            roc = pd.DataFrame({'x': fpr,'y': tpr, 'model': model})
            p = (ggplot(roc, aes(x='x',y='y', color='model'))+geom_line())
        else:
            roc = pd.DataFrame({'x': fpr,'y': tpr})
            p = (ggplot(roc, aes(x='x',y='y'))+geom_line(color='orange'))

        p = (p+geom_abline(intercept = 0, slope = 1, color='blue', linetype='dashed')+
             xlab('False Positive Rate')+ylab('True Positive Rate')+annotate(geom='text',x=0.65,y=0.125,label=anno)+
             ggtitle('Receiver operating characteristic')+scale_x_continuous(expand=(0,0))+scale_y_continuous(expand=(0,0))+theme_bw())
        return p
    elif metric == 'PR':
        recall = x
        precision = y
        aver_precision = anno
        if model is not None:
            pr = pd.DataFrame({'x': recall,'y': precision, 'model': model})
            p = (ggplot(pr, aes(x='x',y='y', color='model'))+geom_line())
        else:
            pr = pd.DataFrame({'x': recall,'y': precision})
            p = (ggplot(pr, aes(x='x',y='y'))+geom_line(color='orange'))
        
        
        p = (p+xlab('False Positive Rate')+ylab('True Positive Rate')+
             annotate(geom='text',x=0.65,y=0.125,label=anno)+ggtitle('Precision-Recall curve')+
             scale_x_continuous(expand=(0,0))+scale_y_continuous(expand=(0,0))+theme_bw())
        return p
    else:
        print('Parameter "metric" should be "ROC" or "PR" not "{}"',format(metric))
        return 0
    
    
def get_metrics(outputs, labels, metric='ROC'):
    if metric == 'ROC':
        fpr, tpr, _ = roc_curve(labels, outputs)
        au_roc = auc(fpr, tpr)
        return fpr, tpr, au_roc
    elif metric == 'PR':
        recall, precision, _ = precision_recall_curve(labels, outputs)
        aver_precision = average_precision_score(labels, outputs)
        return recall, precision, aver_precision
    else:
        print('Parameter "metric" should be "ROC" or "PR" not "{}"',format(metric))
        return 0
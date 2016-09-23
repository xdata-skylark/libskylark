'''
Created on Dec 19, 2012

@author: vikas
'''


def classification_accuracy(Pred, Yt):
        accuracy = sum(Pred==Yt)*100.0/len(Yt)
        return accuracy
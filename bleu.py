'''
    a script that measures the similarity between input
    training text and output inferenced text using bleu score,
    WIP
'''

import re
import nltk.translate.bleu_score as bs


def bleu(ref, gen):
    '''
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for j in gen:
        gen_bleu.append(j.split())
    for i, j in enumerate(ref):
        ref_bleu.append([j.split()])
    cc = bs.SmoothingFunction()
    score_bleu = bs.corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0),
                                smoothing_function=cc.method4)
    return score_bleu


def splice(text):
    '''
    splice(words.txt)
    splits a text file into individual sentences
    '''
    with open(text) as file:
        for line in file:
            for i in re.split(r"(\. |\? |\! )", line):
                string += i
            string += '\n'


splice('abstractsCLEAN.txt')

#wip

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import re

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
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu

def splice(text):
    ''' 
    splice(words.txt)
    '''
    with open(text) as file:
        for line in file:
            for l in re.split(r"(\. |\? |\! )",line):
                string += l
            string += '\n'

splice('abstractsCLEAN.txt')
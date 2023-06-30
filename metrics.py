'''
    a script that measures the similarity between input
    training text and output inferenced text using bleu and rouge scores/

    BLEU focuses on precision: how much the words (and/or n-grams)
    in the candidate model outputs appear in the human reference.

    ROUGE focuses on recall: how much the words (and/or n-grams)
    in the human references appear in the candidate model outputs.

'''

import re
import nltk
import sys
import random
import rouge_metric


def bleu(ref, gen):
    '''
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = [sent.split() for sent in ref]
    gen_bleu = [sent.split() for sent in gen]

    ref_bleu = [random.sample(ref_bleu, 5000) for i in range(len(gen_bleu))]

    cc = nltk.translate.bleu_score.SmoothingFunction()
    # adjust weight
    score_bleu = nltk.translate.bleu_score.corpus_bleu(ref_bleu, gen_bleu, weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=cc.method4)

    return score_bleu

rouge = rouge_metric.PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False, rouge_s=False, rouge_su=False)

def find_my_rouge(text):
    #not used currently
    hypotheses = [[text.split()]]
    score = rouge.evaluate_tokenized(hypotheses, [[ref_sentences]])
    return score

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|vs)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>"
                  * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]"
                  + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]",
                  "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


with open(sys.argv[1], 'r') as file:
    ref = file.read().replace('\n', '')
ref_sentences = split_into_sentences(ref)


with open(sys.argv[2], 'r') as file:
    gen = file.read().replace('\n', '')
gen_sentences = split_into_sentences(gen)

print('BLEU:', bleu(ref_sentences, gen_sentences))
#print('ROUGE:', find_my_rouge(gen_sentences))

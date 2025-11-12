 
# imports 
import string 
import random 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('reuters') 
from nltk.corpus import reuters 
from nltk import FreqDist 
  
# input the reuters sentences 
sents  =reuters.sents() 
  
# write the removal characters such as : Stopwords and punctuation 
stop_words = set(stopwords.words('english')) 
string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—' 
string.punctuation 
removal_list = list(stop_words) + list(string.punctuation)+ ['lt','rt'] 
removal_list 
  
# generate unigrams bigrams trigrams 
unigram=[] 
bigram=[] 
trigram=[] 
tokenized_text=[] 
for sentence in sents: 
  sentence = list(map(lambda x:x.lower(),sentence)) 
  for word in sentence: 
        if word== '.': 
            sentence.remove(word)  
        else: 
            unigram.append(word) 
    
  tokenized_text.append(sentence) 
  bigram.extend(list(ngrams(sentence, 2,pad_left=True, pad_right=True))) 
  trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True))) 
  
# remove the n-grams with removable words 
def remove_stopwords(x):      
    y = [] 
    for pair in x: 
        count = 0
        for word in pair: 
            if word in removal_list: 
                count = count or 0
            else: 
                count = count or 1
        if (count==1): 
            y.append(pair) 
    return (y) 
unigram = remove_stopwords(unigram) 
bigram = remove_stopwords(bigram) 
trigram = remove_stopwords(trigram) 
  
# generate frequency of n-grams  
freq_bi = FreqDist(bigram) 
freq_tri = FreqDist(trigram) 
  
d = defaultdict(Counter) 
for a, b, c in freq_tri: 
    if(a != None and b!= None and c!= None): 
      d[a, b] += freq_tri[a, b, c] 
        
  
# Next word prediction       
s='' 
def pick_word(counter): 
    "Chooses a random element."
    return random.choice(list(counter.elements())) 
prefix = "he", "said"
print(" ".join(prefix)) 
s = " ".join(prefix) 
for i in range(19): 
    suffix = pick_word(d[prefix]) 
    s=s+' '+suffix 
    print(s) 
    prefix = prefix[1], suffix 
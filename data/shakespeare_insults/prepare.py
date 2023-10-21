# a small and rather mean dataset of Shakespearean Insults
import os
from typing import Any, List, Dict, Union
import pickle
import numpy as np

# this dataset comes for FREE inside this repository, that is a pretty good deal!
insults = ["A most notable coward, an infinite and endless liar, an hourly promise breaker, the owner of no one good quality.", "Away, you starvelling, you elf-skin, you dried neat’s-tongue, bull’s-pizzle, you stock-fish!", "Away, you three-inch fool! ", "Come, come, you froward and unable worms!", "Go, prick thy face, and over-red thy fear, Thou lily-liver’d boy.", "His wit’s as thick as a Tewkesbury mustard.", "I am pigeon-liver’d and lack gall.", "I am sick when I do look on thee", "I must tell you friendly in your ear, sell when you can, you are not for all markets", "If thou wilt needs marry, marry a fool; for wise men know well enough what monsters you make of them", "I’ll beat thee, but I would infect my hands", "I scorn you, scurvy companion.", "Methink’st thou art a general offence and every man should beat thee", "More of your conversation would infect my brain", "My wife’s a hobby horse", "Peace, ye fat guts", "Aroint thee: go away, rump-fed runion: slu", "The rankest compound of villainous smell that ever offended nostri", "The tartness of his face sours ripe grapes", "There’s no more faith in thee than in a stewed prune", "Thine forward voice, now, is to speak well of thine friend; thine backward voice is to utter foul speeches and to detract", "That trunk of humours, that bolting-hutch of beastliness, that swollen parcel of dropsies, that huge bombard of sack, that stuffed ", "cloak-bag of guts", "that roasted Manningtree ox with pudding in his belly", "that reverend vice" "Thine face is not worth sunburning", "This woman’s an easy glove, my lord, she goes off and on at pleasure", "Thou art a boil, a plague sor", "Was the Duke a flesh-monger, a fool and a coward", "Thou art as fat as butter", "Here is the babe, as loathsome as a toad", "Like the toad; ugly and venomous", "Thou art unfit for any place but hell", "Thou cream faced loo", "Thou clay-brained guts, thou knotty-pated fool, thou whoreson obscene greasy tallow-catch", "Thou damned and luxurious mountain goat", "Thou elvish-mark’d, abortive, rooting hog", "Thou leathern-jerkin, crystal-button, knot-pated, agatering, puke-stocking, caddis-garter, smooth-tongue, Spanish pouch", "Thou lump of foul deformit", "That poisonous bunch-back’d toad", "Thou sodden-witted lord! Thou hast no more brain than I have in mine elbows", "Thou subtle, perjur’d, false, disloyal man", "Thou whoreson zed , thou unnecessary letter", "Thy sin’s not accidental, but a trade", "Thy tongue outvenoms all the worms of Nile", "Would thou wert clean enough to spit upo", "Would thou wouldst burst", "You poor, base, rascally, cheating lack-linen mate!", "You are as a candle, the better burnt out", "You scullion! You rampallian! You fustilarian! I’ll tickle your catastrophe", "You starvelling, you eel-skin, you dried neat’s-tongue, you bull’s-pizzle, you stock-fish–O for breath to utter what is like ", "Your brain is as dry as the remainder biscuit after voyage", "Virginity breeds mites, much like a cheese", "Villain, I have done thy mothe", "Heaven truly knows that thou art false as hel", "Out of my sight! Thou dost infect mine eyes", "No longer from head to foot than from hip to hip, she is spherical, like a globe; I could find countries in her", "You have such a February face, So full of frost, of storm, and cloudiness"]

n = len(insults)
train_data = insults[:int(n*0.9)]
val_data   = insults[int(n*0.9):]

# NOTE: this was written whilst in Panama!
# here is a simple tokenizer
class InsultsTokenizer:
    """ A tokenizer, specifically conjured from the ether to deal with these insults from shakespeare """
    def __init__(self, dataset: List[str]=None, path: str=None) -> None:
        unique_tokens = set()
        if (path) and (os.path.exists(path)):
            with open(path, "rb") as fileobj:
                tsukemono = pickle.load(fileobj) #instance of pickle!
            if type(tsukemono) == InsultsTokenizer:
                self.__dict__ = tsukemono.__dict__
        # assume import or constructing a new tokenizer for a given dataset
        else:
            assert dataset is not None, "no path provided and no dataset, what could we possibly be tokenizing?"
            for sample in dataset:
                toks = sample.lower().split(" ") #for simplicity sake, keep everything lowercase
                unique_tokens.update(toks)

            unique_tokens = sorted(list(unique_tokens))
            tokmap = {
                "<eos>": 0, #add in an end of sequence token, model can decide to stop frying people
                " ": 1,     #add in the space!
                "<wtf>": 2, #for out of distribution tokens, lord knows what these might be
            }

            # Special Tokens to use based on their meaning
            self.eos_token_id = 0
            self.space_token_id = 1
            self.ood_token_id = 2

            token_idx = len(tokmap) #number @ which we should add new tokens (after special toks)
            for tok in unique_tokens:
                tokmap[tok] = token_idx
                token_idx += 1

            self.encoder_map: Dict[str, int] = tokmap
            self.decoder_map: Dict[str, int] = {tok: char for char, tok in self.encoder_map.items()}
            self.vocab_dim = len(tokmap.keys())
            del tokmap
            self.max_seq_len = 64 #manually looked @ dataset, set this num to allow for some leeway at inference time

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> Any:
        text = text.lower() #keep everything lowercase here
        text_chunks = text.split(" ")
        tokens: List[int] = [] # our returnable

        for chunk in text_chunks:
            if chunk in self.encoder_map:
                chunk_toks = [self.encoder_map[chunk], self.space_token_id]
            else:
                chunk_toks = [self.ood_token_id, self.space_token_id]
            tokens.extend(chunk_toks)
        tokens = tokens[:-1] #remove the last extra space

        # to teach the model to STOP espousing insults, we include a special end of sequence token
        if ("pad" in kwargs) and (kwargs["pad"] == True):
            remaining_seq_len = self.max_seq_len - len(tokens)
            eos_seq = remaining_seq_len * [self.eos_token_id]
            tokens.extend(eos_seq)

        return tokens

    def decode(self, tokens: Union[List[List[int]], List[int]]) -> Union[List[str], str]:
        """ Take tokens, convert into original strings """
        decoded = None
        # Batch of tokens to decode 
        if type(tokens[0]) == list:
            decoded = []
            for sequence in tokens:
                decoded.append([self.decoder_map[token_id] for token_id in sequence])
        # Single list of tokens to decode
        elif (type(tokens) == list and type(tokens[0]) == int):
            decoded = [self.decoder_map[token_id] for token_id in tokens]
        else:
            raise NotImplementedError("an unsupported combination of dtypes was passed in to 'decode()' for 'tokens'")
        return decoded

    def save(self, path: str=None):
        """ dump the tokenizer to a path on disk """
        assert path is not None
        with open(path, "wb") as tsukemono:
            pickle.dump(self, tsukemono)


# Dump this thang to disk
tokenizer = InsultsTokenizer(dataset=insults)

# :::: Sample Usage of the Tokenizer ::::
# tokenizing a string
# s ="what in tarntation you whoreson" 
# toks = tokenizer(s)
# print(toks)
# print(len(toks))
# string = tokenizer.decode(toks)

# saving to disk
# tmp_path = "tmp.pkl"
# tokenizer.save(tmp_path)

# loading from disk on init
# t2 = InsultsTokenizer(path=tmp_path)


# Tokenize the datasets and save the splits
train_ids = [tokenizer(sample, pad=True) for sample in train_data]
total_training_toks = sum([len(i) for i in train_ids])
val_ids = [tokenizer(sample, pad=True) for sample in val_data]
total_val_toks = sum([len(i) for i in val_ids])
print(f"train has {len(train_ids):,} instances with {total_training_toks} tokens")
print(f"val has {len(val_ids):,} instances with {total_val_toks} tokens")


# FIND THE MAX SEQLEN 
results = map(tokenizer.__call__, insults)
m = 0
for i in results:
    # print(len(i))
    m = max(m, len(i))
print("longest seqlen in dataset: ", m, "w/o padding...")
print(f"vocabulary size: {tokenizer.vocab_dim}")
    

# export the tokenized data to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, for building and/or reading models specifically for this dataset
meta = {
    'vocab_size': tokenizer.vocab_dim,
    'itos': tokenizer.encoder_map,
    'stoi': tokenizer.decoder_map,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

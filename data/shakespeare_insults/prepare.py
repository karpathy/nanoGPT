# a rather small dataset of Shakespeare-an Insults
import os
import requests
import tiktoken
import numpy as np

# it comes for FREE inside this repository
insults = ["A most notable coward, an infinite and endless liar, an hourly promise breaker, the owner of no one good quality.", "Away, you starvelling, you elf-skin, you dried neat’s-tongue, bull’s-pizzle, you stock-fish!", "Away, you three-inch fool! ", "Come, come, you froward and unable worms!", "Go, prick thy face, and over-red thy fear, Thou lily-liver’d boy.", "His wit’s as thick as a Tewkesbury mustard.", "I am pigeon-liver’d and lack gall.", "I am sick when I do look on thee", "I must tell you friendly in your ear, sell when you can, you are not for all markets", "If thou wilt needs marry, marry a fool; for wise men know well enough what monsters you make of them", "I’ll beat thee, but I would infect my hands", "I scorn you, scurvy companion.", "Methink’st thou art a general offence and every man should beat thee", "More of your conversation would infect my brain", "My wife’s a hobby horse", "Peace, ye fat guts", "Aroint thee: go away, rump-fed runion: slu", "The rankest compound of villainous smell that ever offended nostri", "The tartness of his face sours ripe grapes", "There’s no more faith in thee than in a stewed prune", "Thine forward voice, now, is to speak well of thine friend; thine backward voice is to utter foul speeches and to detract", "That trunk of humours, that bolting-hutch of beastliness, that swollen parcel of dropsies, that huge bombard of sack, that stuffed ", "cloak-bag of guts", "that roasted Manningtree ox with pudding in his belly", "that reverend vice" "Thine face is not worth sunburning", "This woman’s an easy glove, my lord, she goes off and on at pleasure", "Thou art a boil, a plague sor", "Was the Duke a flesh-monger, a fool and a coward", "Thou art as fat as butter", "Here is the babe, as loathsome as a toad", "Like the toad; ugly and venomous", "Thou art unfit for any place but hell", "Thou cream faced loo", "Thou clay-brained guts, thou knotty-pated fool, thou whoreson obscene greasy tallow-catch", "Thou damned and luxurious mountain goat", "Thou elvish-mark’d, abortive, rooting hog", "Thou leathern-jerkin, crystal-button, knot-pated, agatering, puke-stocking, caddis-garter, smooth-tongue, Spanish pouch", "Thou lump of foul deformit", "That poisonous bunch-back’d toad", "Thou sodden-witted lord! Thou hast no more brain than I have in mine elbows", "Thou subtle, perjur’d, false, disloyal man", "Thou whoreson zed , thou unnecessary letter", "Thy sin’s not accidental, but a trade", "Thy tongue outvenoms all the worms of Nile", "Would thou wert clean enough to spit upo", "Would thou wouldst burst", "You poor, base, rascally, cheating lack-linen mate!", "You are as a candle, the better burnt out", "You scullion! You rampallian! You fustilarian! I’ll tickle your catastrophe", "You starvelling, you eel-skin, you dried neat’s-tongue, you bull’s-pizzle, you stock-fish–O for breath to utter what is like ", "Your brain is as dry as the remainder biscuit after voyage", "Virginity breeds mites, much like a cheese", "Villain, I have done thy mothe", "Heaven truly knows that thou art false as hel", "Out of my sight! Thou dost infect mine eyes", "No longer from head to foot than from hip to hip, she is spherical, like a globe; I could find countries in her", "You have such a February face, So full of frost, of storm, and cloudiness"]

n = len(insults)
train_data = insults[:int(n*0.9)]
val_data   = insults[int(n*0.9):]

# TODO: implement a custom tokenizer, based on the distribution of words inside this toy dataset
# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# TODO: get the token distributions
# train.bin has N tokens
# val.bin has N tokens

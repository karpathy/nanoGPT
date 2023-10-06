import bitstring
import random 

span = 10000000
iteration = 100000

def ieee754(flt):
    b = bitstring.BitArray(float=flt, length=32)
    return b

with open("TestVectorMultiply", "w") as f:

    for i in range(iteration):
        a = ieee754(random.uniform(-span, span))
        b = ieee754(random.uniform(-span, span))
        ab = ieee754(a.float * b.float)

        f.write(a.hex +"_" +  b.hex  +  "_" + ab.hex + "\n")



##############END OF PROGRAM###########################################################
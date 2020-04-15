# DNA fragile and flexibility scores are used from BreakSeq, Hugo et al. 2010

flextab = \
{"AA":7.6 , "CA":10.9, "GA":8.8 , "TA":12.5,
 "AC":14.6, "CC":7.2 , "GC":11.1, "TC":8.8,
 "AG":8.2 , "CG":8.9 , "GG":7.2 , "TG":10.9,
 "AT":25.0, "CT":8.2 , "GT":14.6, "TT":7.6}

helixtab = \
{"AA":1.9,  "CA":1.9,  "GA":1.6,  "TA":0.9,
 "AC":1.3,  "CC":3.1,  "GC":3.1,  "TC":1.6,
 "AG":1.6,  "CG":3.6,  "GG":3.1,  "TG":1.9,
 "AT":1.5,  "CT":1.6,  "GT":1.3,  "TT":1.9}

def calc_fragility(sequence, dinucleotides):
	frag = 0
	total = 0
	for i in range(len(sequence)-1):
		dint = str(sequence[i:i+2]).upper()
		if dint in dinucleotides:
			frag = frag + dinucleotides[dint]
			total = total + 1
	return frag/total if total > 0 else 0

def calc_flexibility(sequence):
	return calc_fragility(sequence, flextab)

def calc_stability(sequence):
	return calc_fragility(sequence, helixtab)

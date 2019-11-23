
disaster_vocab = []

filename = "CrisisLexRec.txt"
with open(filename) as infile:
    for line in infile:
        phrase = line.strip('\n').strip(' ').strip('').strip('\t').lower()
        if phrase != "":
            disaster_vocab.append(phrase)

filename = "Lexicon.txt"
with open(filename) as infile:
    for line in infile:
        phrase = line.strip('\n').strip(' ').strip('').strip('\t').lower()
        if phrase != "":
            disaster_vocab.append(phrase)


disaster_vocab = list(set(disaster_vocab))
disaster_vocab.sort()

print(len(disaster_vocab))

filename = "Lexicon.txt"

with open(filename, "w+") as infile:
    for word in disaster_vocab:
        infile.write(word+"\n")

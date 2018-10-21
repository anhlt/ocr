
class Lang:
    def __init__(self):
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_characters = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for char in sentence:
            self.addChar(char)

    def addChar(self, character):
        if character not in self.char2index:
            self.char2index[character] = self.n_characters
            self.char2count[character] = 1
            self.index2char[self.n_characters] = character
            self.n_characters += 1
        else:
            self.char2count[character] += 1

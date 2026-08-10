class SentenceBuilder:

    def __init__(self):
        self.current_word = ""
        self.sentence = []

    def add_letter(self, letter):
        """Add a predicted ASL letter to the current word."""
        self.current_word += letter

    def add_space(self):
        """Finish the current word and add it to the sentence."""
        if self.current_word:
            self.sentence.append(self.current_word)
            self.current_word = ""

    def backspace(self):
        """Remove the most recently added letter."""
        if self.current_word:
            self.current_word = self.current_word[:-1]

    def get_current_word(self):
        return self.current_word

    def get_sentence(self):
        return " ".join(self.sentence)

    def reset(self):
        """Clear the entire sentence."""
        self.current_word = ""
        self.sentence = []
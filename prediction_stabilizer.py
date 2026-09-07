from collections import Counter, deque


class PredictionStabilizer:
    """
    Temporal stabilizer for ASL predictions.

    A prediction is accepted only when:
      1. There are enough recent predictions.
      2. One class has a strong majority.
      3. The winning class has sufficient confidence.
      4. The same letter cannot immediately be accepted again.

    This prevents one letter from being repeatedly appended
    while the user is holding the same sign.
    """

    def __init__(
        self,
        window_size=12,
        min_confidence=0.30,
        min_votes=8,
        min_average_confidence=0.40,
    ):

        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_votes = min_votes
        self.min_average_confidence = min_average_confidence

        self.history = deque(
            maxlen=window_size
        )

        self.locked_letter = None


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        prediction,
        confidence,
    ):

        # Ignore very weak predictions.
        if confidence < self.min_confidence:
            return None

        self.history.append(
            (prediction, confidence)
        )

        # Wait until we have enough temporal information.
        if len(self.history) < self.window_size:
            return None


        # ----------------------------------------------------
        # Count votes
        # ----------------------------------------------------

        votes = Counter(
            letter
            for letter, _ in self.history
        )

        best_letter, best_votes = (
            votes.most_common(1)[0]
        )


        # ----------------------------------------------------
        # Confidence of winning class
        # ----------------------------------------------------

        winning_confidences = [
            confidence
            for letter, confidence
            in self.history
            if letter == best_letter
        ]

        average_confidence = (
            sum(winning_confidences)
            / len(winning_confidences)
        )


        # ----------------------------------------------------
        # Require strong majority
        # ----------------------------------------------------

        if best_votes < self.min_votes:
            return None

        if average_confidence < self.min_average_confidence:
            return None


        # ----------------------------------------------------
        # Don't repeatedly emit same letter
        # ----------------------------------------------------

        if self.locked_letter == best_letter:
            return None


        # ----------------------------------------------------
        # Accept
        # ----------------------------------------------------

        self.locked_letter = best_letter

        return best_letter


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.history.clear()

        self.locked_letter = None
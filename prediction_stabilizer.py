from collections import Counter


class PredictionStabilizer:

    def __init__(self, required_frames=8):
        self.required_frames = required_frames
        self.predictions = []
        self.last_accepted = None

    def update(self, prediction):
        """
        Returns a letter only once when it becomes stable.

        The same letter will not be accepted again until
        a different prediction becomes stable.
        """

        self.predictions.append(prediction)

        # Keep only the most recent predictions
        if len(self.predictions) > self.required_frames:
            self.predictions.pop(0)

        # Not enough frames yet
        if len(self.predictions) < self.required_frames:
            return None

        # Find the most common prediction
        most_common, count = Counter(
            self.predictions
        ).most_common(1)[0]

        # Require all frames to agree
        if count == self.required_frames:

            # Don't accept the same gesture repeatedly
            if most_common == self.last_accepted:
                return None

            # New stable letter
            self.last_accepted = most_common
            self.predictions.clear()

            return most_common

        return None

    def reset(self):
        self.predictions.clear()
        self.last_accepted = None


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    stabilizer = PredictionStabilizer(
        required_frames=8
    )

    predictions = [
        # Hold H
        "H", "H", "H", "H",
        "H", "H", "H", "H",

        # Continue holding H
        "H", "H", "H", "H",
        "H", "H", "H", "H",

        # Change to E
        "E", "E", "E", "E",
        "E", "E", "E", "E",

        # Continue holding E
        "E", "E", "E", "E",
        "E", "E", "E", "E",
    ]

    for prediction in predictions:

        result = stabilizer.update(
            prediction
        )

        if result:
            print("Accepted:", result)
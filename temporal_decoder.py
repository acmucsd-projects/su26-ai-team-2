from collections import deque, Counter
import numpy as np


class TemporalPredictionDecoder:
    """
    Temporal decoder for ASL sign recognition.

    Behavior:
        - Requires multiple consistent predictions before accepting a sign.
        - Uses confidence, margin, and consensus checks.
        - Suppresses duplicate acceptance while the same sign is held.
        - A no-hand/reset event unlocks the decoder so the same sign
          can be accepted again after the hand leaves the frame.

    Example:

        A held        -> A
        A held        -> nothing
        A held        -> nothing
        no hand       -> reset
        A again       -> A
    """

    def __init__(
        self,
        window_size=15,
        min_stable_predictions=8,
        min_confidence=0.45,
        min_margin=0.15,
        min_consensus_ratio=0.60,
        suppress_duplicate=True,
        recency_weight=1.15,
    ):
        self.window_size = window_size
        self.min_stable_predictions = min_stable_predictions
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.min_consensus_ratio = min_consensus_ratio
        self.suppress_duplicate = suppress_duplicate
        self.recency_weight = recency_weight

        self.prediction_history = deque(maxlen=window_size)
        self.probability_history = deque(maxlen=window_size)

        # Last sign that was accepted.
        self.last_accepted_label = None

        # Whether the decoder is currently allowed to emit a sign.
        #
        # False immediately after accepting a sign.
        # True again after reset/no-hand.
        self.ready_for_new_sign = True

    def update(self, prediction, probabilities=None):
        """
        Add a new prediction to the temporal window.

        Parameters
        ----------
        prediction : str
            Current raw predicted class.

        probabilities : array-like, optional
            Full CNN probability vector.

        Returns
        -------
        str or None
            Accepted sign, or None if the prediction should not
            yet be emitted.
        """

        if prediction is None:
            return None

        prediction = str(prediction)

        self.prediction_history.append(prediction)

        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=float)
            self.probability_history.append(probabilities)

        # Need enough temporal observations.
        if len(self.prediction_history) < self.min_stable_predictions:
            return None

        # ---------------------------------------------------------
        # 1. Determine majority prediction
        # ---------------------------------------------------------

        counts = Counter(self.prediction_history)

        majority_label, majority_count = counts.most_common(1)[0]

        consensus_ratio = majority_count / len(self.prediction_history)

        if consensus_ratio < self.min_consensus_ratio:
            return None

        # ---------------------------------------------------------
        # 2. Calculate temporal confidence/margin
        # ---------------------------------------------------------

        temporal_confidence = None
        temporal_margin = None

        if len(self.probability_history) > 0:

            probs = list(self.probability_history)

            weights = np.array(
                [
                    self.recency_weight ** i
                    for i in range(len(probs))
                ],
                dtype=float,
            )

            weights /= weights.sum()

            temporal_probs = np.average(
                np.array(probs),
                axis=0,
                weights=weights,
            )

            sorted_probs = np.sort(temporal_probs)

            temporal_confidence = float(sorted_probs[-1])

            if len(sorted_probs) >= 2:
                temporal_margin = float(
                    sorted_probs[-1] - sorted_probs[-2]
                )
            else:
                temporal_margin = temporal_confidence

            # Find class index corresponding to majority label.
            #
            # The caller's label mapping is not necessarily available
            # here, so confidence is checked against the temporal
            # winning class rather than assuming a specific index.
            temporal_winner_index = int(np.argmax(temporal_probs))

            # If the caller supplied probabilities but the raw majority
            # label does not correspond to the temporal winner, don't
            # accept yet.
            #
            # We cannot map index -> label here, so this consistency
            # check is intentionally handled primarily through the
            # prediction history.
            if temporal_confidence < self.min_confidence:
                return None

            if temporal_margin < self.min_margin:
                return None

        # ---------------------------------------------------------
        # 3. Duplicate/sign-cycle protection
        # ---------------------------------------------------------

        if self.suppress_duplicate:

            # Same sign is still being held.
            #
            # Do NOT emit it again until reset() occurs.
            if (
                self.last_accepted_label == majority_label
                and not self.ready_for_new_sign
            ):
                return None

        # ---------------------------------------------------------
        # 4. Accept sign
        # ---------------------------------------------------------

        accepted_label = majority_label

        self.last_accepted_label = accepted_label

        # Lock decoder until a reset/no-hand event.
        self.ready_for_new_sign = False

        print()
        print("=" * 50)
        print(f"ACCEPTED LETTER: {accepted_label}")

        if temporal_confidence is not None:
            print(f"Confidence: {temporal_confidence:.3f}")

        if temporal_margin is not None:
            print(f"Margin: {temporal_margin:.3f}")

        print(f"Consensus: {consensus_ratio:.3f}")
        print("=" * 50)

        # Clear the temporal window after acceptance.
        #
        # This prevents the previous sign from dominating the next sign.
        self.prediction_history.clear()
        self.probability_history.clear()

        return accepted_label

    def reset(self):
        """
        Reset temporal state when the hand disappears.

        IMPORTANT:
        This also clears the duplicate lock.

        Therefore:

            A -> accepted
            no hand -> reset()
            A -> accepted again
        """

        self.prediction_history.clear()
        self.probability_history.clear()

        # This is the important sign-cycle correction.
        self.last_accepted_label = None

        # Allow the next sign to be accepted.
        self.ready_for_new_sign = True

    def clear_history(self):
        """
        Clear only temporal history.

        This is different from reset().

        Use this when you want to clear current observations
        without necessarily changing the sign-cycle state.
        """

        self.prediction_history.clear()
        self.probability_history.clear()

    def get_state(self):
        """
        Useful for debugging.
        """

        return {
            "history_size": len(self.prediction_history),
            "last_accepted_label": self.last_accepted_label,
            "ready_for_new_sign": self.ready_for_new_sign,
        }
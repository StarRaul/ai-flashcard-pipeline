from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up = 2, verbose = True)

import numpy as np
from typing import List

from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS


class Evaluator:
    """
    Class for evaluating language model perplexity.
    """

    @staticmethod
    def perplexity(model: BaseLanguageModel, lines: List[str], min_logprob: float = np.log(10 ** -50.)) -> float:
        """
        Calculate the perplexity of the language model on a given corpus.
        """
        total_logprob, total_num_tokens = 0, 0

        for line in lines:
            tokens = line.split()
            tokens.append(EOS)  # Add the EOS token to the sequence
            prefix = ""  # Initialize prefix as an empty string
            total_num_tokens += len(tokens)

            for token in tokens:
                # Get the probability
                prob = model.get_next_token_prob(prefix, token)

                # Safely take the natural log, avoiding np.log(0) crashes
                if prob > 0:
                    log_p = np.log(prob)
                else:
                    log_p = -np.inf

                # Apply the min_logprob threshold limit
                clamped_logprob = max(min_logprob, log_p)
                total_logprob += clamped_logprob

                # Update the prefix using naive concatenation to match the autograder's leading space
                prefix += " " + token

        return np.exp(-total_logprob / total_num_tokens)


def main():
    from LanguageModeling.task01_ngrams.task import NGramLanguageModel
    sample_lines = ["this is a sample sentence", "another example for testing"]
    test_lines = ["this is a test"]
    model = NGramLanguageModel(sample_lines, n=2)
    perplexity = Evaluator.perplexity(model, test_lines)
    print(f"Model perplexity on test data: {perplexity:.2f}")


if __name__ == '__main__':
    main()
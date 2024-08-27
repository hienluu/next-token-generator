from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

class LanguageModel:
    def __init__(self, model_name):
        """
        Initialize the LanguageModel class.

        Args:
        model_name (str): The name of the pre-trained language model to use.
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def getNextTokenProbabilities(self, prompt, topK):
        """
        Get the next token probabilities for a given prompt.

        Args:
        prompt (str): The input prompt.
        topK (int): The number of top probabilities to return.

        Returns:
        A dictionary with the topK next token probabilities.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        top_probabilities = torch.topk(probabilities, topK)
        
        top_tokens = self.tokenizer.convert_ids_to_tokens(top_probabilities.indices[0])
        top_tokens = [token.lstrip("Ġ") for token in top_tokens]
        next_token_probabilities = dict(zip(top_tokens, top_probabilities.values[0].tolist()))
        
        sorted_next_token_probabilities = dict(sorted(next_token_probabilities.items(), key=lambda item: item[1], reverse=True))
        return sorted_next_token_probabilities
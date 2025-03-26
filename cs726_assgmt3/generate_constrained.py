import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class ConstrainedTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        eos_id: int, 
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the ConstrainedTextGenerator class.
            
            model: LLM
            tokenizer: LLM's tokenizer.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"], word_list: list
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Word-Constrained decoding technique. (refer assignment document for more details)
            
            `word_list`: contains bag of words for the particular example

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        trie = {}
        for word in word_list:
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            current = trie
            for token in word_tokens:
                if token not in current:
                    current[token] = {}
                current = current[token]
            current['_end_'] = True
        generated_tokens = []
        current_input_ids = input_ids.clone()
        used_words = set()
        past_key_values = None
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input_ids, use_cache=True, past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            next_token_id = self._get_next_token(logits, trie, used_words, generated_tokens, word_list)
            if next_token_id == self.eos_token_id:
                break
            generated_tokens.append(next_token_id)
            current_input_ids = torch.tensor([[next_token_id]], device=current_input_ids.device)
        return torch.tensor(generated_tokens)
    

    def _get_next_token(self, logits, trie, used_words, generated_tokens, word_list):
        current_trie = trie
        for token in generated_tokens:
            if token in current_trie:
                current_trie = current_trie[token]
            else:
                current_trie = trie
                if token in current_trie:
                    current_trie = current_trie[token]
        if '_end_' in current_trie and current_trie['_end_']:
            current_trie = trie
        mask = torch.full_like(logits, float('-inf'))
        valid_tokens = set()
        for token_id in current_trie.keys():
            if token_id != "_end_":
                valid_tokens.add(token_id)
        valid_tokens.add(self.eos_token_id)
        for token_id in valid_tokens:
            mask[0, token_id] = logits[0, token_id]
        probs = torch.softmax(mask, dim=-1)
        return torch.argmax(probs, dim=-1).item()
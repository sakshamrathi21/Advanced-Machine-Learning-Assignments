import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class TextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        max_output_len: int = 10,
        tau: int = 1,
        k: int = 10,
        p: int = 0.5,
        tokenizer=None
    ) -> None:
        '''
            Initialize the TextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            max_output_len: Maximum number of tokens to be generated.
            tau: Temperature parameter for random sampling
            k: Top-k parameter for top-k sampling
            p: Cumulative probability threshold for nucleus sampling
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        self.tokenizer = tokenizer
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.tau = tau
        self.k = k 
        self.p = p
        
        if decoding_strategy == "greedy":
            self.generator_func = self.greedy_decoding
        elif decoding_strategy == "random":
            self.generator_func = self.random_sampling
        elif decoding_strategy == "topk":
            self.generator_func = self.topk_sampling
        elif decoding_strategy == "nucleus":
            self.generator_func = self.nucleus_sampling

    def __call__(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"], 
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def greedy_decoding(
        self,
        input_ids: Int[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]: 
            generated_tokens = []
            current_ids = input_ids
            past_key_values = None
            for i in range(self.max_output_len):
                print("Step: ", i)
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                logit_last_token = logits[:, -1, :]
                next_token = torch.argmax(logit_last_token, dim=-1)
                token_id = next_token.item()
                generated_tokens.append(token_id)
                if self.tokenizer:
                    token_str = self.tokenizer.decode(token_id)
                    print(f"Generated Token: {token_str}")
                    if i % 10 == 0:
                        full_sequence = self.tokenizer.decode(generated_tokens)
                        print(f"Generated so far: {full_sequence}")
                if token_id == self.eos_token_id:
                    break
                current_ids = next_token.unsqueeze(0)
            
            return torch.tensor(generated_tokens, dtype=torch.long)
        
    def random_sampling(
    self, 
    input_ids: Int[torch.Tensor, "batch in_seq_len"]
) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Random sampling technique.
            
            Sample from the probability distribution after applying temperature.
        '''    
        generated_tokens = []
        current_input_ids = input_ids.clone()
        past_key_values = None
        
        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
            
            next_token_logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values
            next_token_logits = next_token_logits / self.tau
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)
            if self.tokenizer:
                token_str = self.tokenizer.decode(next_token_id)
                print(f"Generated Token: {token_str}")
                if len(generated_tokens) % 10 == 0:
                    full_sequence = self.tokenizer.decode(generated_tokens)
                    print(f"Generated so far: {full_sequence}")
            if next_token_id == self.eos_token_id:
                break
            current_input_ids = next_token.unsqueeze(0)
        return torch.tensor(generated_tokens, dtype=torch.long)
    
    def topk_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Top-k sampling technique. (refer assignment document for more details)

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
        raise NotImplementedError
    
    def nucleus_sampling(
        self, 
        input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Implement Nucleus sampling technique. (refer assignment document for more details)

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
        raise NotImplementedError
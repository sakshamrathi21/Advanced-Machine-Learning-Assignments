import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

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
        current_input = input_ids.clone()
        generated_tokens = []

        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(current_input)
                logits = outputs.logits

            last_token_logits = logits[0, -1, :]
            next_token = torch.argmax(last_token_logits).unsqueeze(0)
            generated_tokens.append(next_token.item())

            if next_token.item() == self.eos_token_id:
                break

            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)

        return torch.tensor(generated_tokens, dtype=torch.long)

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

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
        current_input = input_ids.clone()
        generated_tokens = []
        while len(generated_tokens) < self.max_output_len:
            with torch.no_grad():
                outputs = self.model(current_input, output_orig=True, medusa_forward=True)
                head_log_probs = [torch.log_softmax(outputs[2][0, -1, :], dim=-1)]
                for head_idx in range(0, self.no_heads):
                    head_log_probs.append(torch.log_softmax(outputs[0][head_idx][0, -1, :], dim=-1))
                
            candidates = [current_input.clone()]
            scores = [0.0]

            for s, log_prob_dist in enumerate(head_log_probs):
                new_candidates = []
                new_scores = []
                for c, candidate in enumerate(candidates):
                    top_tokens = torch.topk(log_prob_dist, self.beam_width).indices
                    for top_token in top_tokens:
                        new_candidate = torch.cat([candidate, top_token.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = scores[c] + log_prob_dist[top_token].item()
                        new_candidates.append(new_candidate)
                        new_scores.append(new_score)
                if len(new_candidates) > 0:
                    top_indices = torch.topk(torch.tensor(new_scores), min(self.beam_width, len(new_scores))).indices
                    candidates = [new_candidates[i] for i in top_indices]
                    scores = [new_scores[i] for i in top_indices]
            final_scores = []
            for candidate in candidates:
                with torch.no_grad():
                    candidate_outputs = self.model(candidate)
                    candidate_logits = candidate_outputs.logits
                    candidate_score = 0
                    for t in range(current_input.shape[1], candidate.shape[1]):
                        token_logits = torch.softmax(candidate_logits[0, t-1, :], dim=-1)
                        candidate_score += token_logits[candidate[0, t].item()]
                    final_scores.append(candidate_score.item())
            best_candidate_idx = torch.argmax(torch.tensor(final_scores)).item()
            best_candidate = candidates[best_candidate_idx]
            for t in range(current_input.shape[1], best_candidate.shape[1]):
                next_token = best_candidate[0, t].item()
                generated_tokens.append(next_token)
                if next_token == self.eos_token_id:
                    return torch.tensor(generated_tokens, dtype=torch.long)
            current_input = best_candidate
            
        return torch.tensor(generated_tokens, dtype=torch.long)
            
from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        # if load_tokenizer:
        #     self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        # if load_actor_critic:
        #     self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs(self, obs: torch.ByteTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    def act_transformer(self, obs: torch.ByteTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        assert self.keys_values_wm is not None
        # should first refresh_keys_values_with_initial_obs
        obs = obs.float()
        B = obs.size(0)
        L = obs.size(1)
        obs = obs.view(B*L, obs.size(2), obs.size(3), obs.size(4))
        obs_vector = self.vectoriser(obs)
        obs_vectors = obs_vector.view(B, L, self.world_model.config.tokens_per_block-1, self.world_model.vec_size)
        obs_vectors_mapped = self.obs_map(obs_vectors)
        sequences = obs_vectors_mapped.view(B, L*self.world_model.config.tokens_per_block, self.world_model.embed_dim)

        # TODO: should pass obs vectors one by one?
        outputs = self.world_model(sequences, past_keys_values=self.keys_values_wm)

        logits_actions = outputs.logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:

        return self.act_transformer(obs, should_sample, temperature)
        
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

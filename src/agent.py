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
        return self.world_model.transformer.ln_f.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        # if load_tokenizer:
        #     self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        # if load_actor_critic:
        #     self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def reset(self, obs: torch.ByteTensor, actions: torch.LongTensor, batch_size: int = None):
        assert batch_size is not None or obs is not None
        self.prev_steps = 0
        if obs is None:
            self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=batch_size, max_tokens=self.world_model.config.max_tokens)
        if obs is not None:
            self.refresh_keys_values_with_initial_obs(obs, actions)

    def clear(self):
        self.keys_values_wm = None
        self.prev_steps = 0

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs(self, obs: torch.ByteTensor, actions: torch.LongTensor) -> torch.FloatTensor:
        # n, num_observations_tokens = obs.shape
        n, L = obs.shape[0: 2]
        # assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)

        B = n
        obs_vectors_mapped = self.world_model.obs_to_vectors(obs)

        if actions is None:
            sequences = obs_vectors_mapped.view(B, L*self.world_model.config.tokens_per_block-1, self.world_model.embed_dim)
        else:
            assert obs_vectors_mapped.ndim == 4
            # obs_vecs: B, L, N(16), E(256)
            act_vec = self.world_model.act_embedder(actions) # shape: B, L, D
            act_vec = act_vec.view(B, L, 1, self.world_model.embed_dim)
            sequences = torch.cat((obs_vectors_mapped, act_vec), dim=2)
            sequences = sequences.view(B, L*self.world_model.config.tokens_per_block, self.world_model.embed_dim)

        assert self.prev_steps == 0
        outputs_wm = self.world_model(sequences, past_keys_values=self.keys_values_wm, prev_steps=self.prev_steps)
        self.prev_steps += sequences.size(1)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def act_transformer(self, obs: torch.ByteTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        # print('>>> TODO: add history sequence for collector')
        # assert self.keys_values_wm is not None
        # should first refresh_keys_values_with_initial_obs
        obs = obs.unsqueeze(1)
        B = obs.size(0)
        L = obs.size(1)
        obs_vectors_mapped = self.world_model.obs_to_vectors(obs)
        sequences = obs_vectors_mapped.view(B, L*self.world_model.config.tokens_per_block-1, self.world_model.embed_dim)

        # TODO: should pass obs vectors one by one?
        outputs = self.world_model(sequences, past_keys_values=self.keys_values_wm, prev_steps=self.prev_steps)
        self.prev_steps += sequences.size(1)

        logits_actions = outputs.logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)

        # update kv_cache for action token
        actions = act_token.view(B, L)
        act_vec = self.world_model.actions_to_vectors(actions)

        sequences = act_vec.view(B, 1, self.world_model.embed_dim)

        _ = self.world_model(sequences, past_keys_values=self.keys_values_wm, prev_steps=self.prev_steps)
        self.prev_steps += sequences.size(1)

        return act_token

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:

        return self.act_transformer(obs, should_sample, temperature)

        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token

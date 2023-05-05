from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .vectoriser import Vectoriser
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses, compute_lambda_returns
from torch.distributions.categorical import Categorical


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        self.vectoriser = Vectoriser()
        self.vec_size = 512 // (self.config.tokens_per_block - 1)
        self.obs_map = nn.Linear(self.vec_size, config.embed_dim)

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0 # why but last obs, a walkaround for actor_critic.imagine
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.embed_dim = config.embed_dim

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        self.act_embedder = nn.Embedding(act_vocab_size, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([self.act_embedder, nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            # block_mask=all_but_last_obs_tokens_pattern,
            # correspondingly use 'else' in world_model_env:step()
            block_mask=obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                # nn.Linear(config.embed_dim, obs_vocab_size)
                nn.Linear(config.embed_dim, config.embed_dim)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        self.head_actions = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, act_vocab_size)
            )
        )

        self.head_values = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 1)
            )
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    # def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:
    def forward(self, sequences: torch.FloatTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        # num_steps = tokens.size(1)  # (B, T)
        num_steps = sequences.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_actions = self.head_actions(x, num_steps=num_steps, prev_steps=prev_steps)
        values = self.head_values(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_actions, values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        return self.compute_loss_rl(batch, **kwargs)

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        # token: (b, seq_len*(K+1))
        outputs = self(tokens)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)
            , 'b t k -> b (t k)')[:, 1:]
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)

    def compute_labels_world_model_rl(self, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_rewards.reshape(-1), labels_ends.reshape(-1)

    def compute_loss_rl(self, batch: Batch, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        # with torch.no_grad():
        #     obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        obs = batch['observations'].float()
        B = obs.size(0)
        L = obs.size(1)
        obs = obs.view(B*L, obs.size(2), obs.size(3), obs.size(4))
        obs_vector = self.vectoriser(obs)

        obs_vectors = obs_vector.view(B, L, self.config.tokens_per_block-1, self.vec_size)

        obs_vectors_mapped = self.obs_map(obs_vectors)

        act_tokens = batch['actions']
        act_vec = self.act_embedder(act_tokens) # shape: B, L, D
        act_vec = act_vec.view(B, L, 1, self.embed_dim)

        sequences = torch.cat((obs_vectors_mapped, act_vec), dim=2)
        sequences = sequences.view(B, L*self.config.tokens_per_block, self.embed_dim)

        # act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        # tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        # >>> DEBUG log obs torch.Size([10, 320, 512])
        # >>> DEBUG mask torch.Size([10, 20])
        # >>> DEBUG rewards torch.Size([10, 20])
        # >>> DEBUG ends torch.Size([10, 20])

        outputs = self(sequences)

        labels_rewards, labels_ends = self.compute_labels_world_model_rl(batch['rewards'], batch['ends'], batch['mask_padding'])

        # logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        logits_observations = outputs.logits_observations[:, :-1]
        labels_observations = outputs.logits_observations[:, 1:]

        mask_padding = batch['mask_padding'] # shape: B, L
        # loss_obs = (labels_observations - logits_observations).pow(2).mean(dim=-1)
        loss_obs = 1 - F.cosine_similarity(logits_observations, labels_observations, dim=-1)

        mask_fill = torch.logical_not(mask_padding)
        mask_fill = mask_fill.unsqueeze(-1).expand(-1, -1, self.config.tokens_per_block-1)
        mask_fill = mask_fill.reshape(B, -1)[:, :-1]

        loss_obs = loss_obs.masked_fill(mask_fill, 0)
        loss_obs = loss_obs.sum() / mask_padding.int().sum()

        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        values = outputs.values.squeeze(-1)

        # RL loss
        lambda_returns = compute_lambda_returns(
            rewards=batch['rewards'],
            values=values,
            ends=batch['ends'],
            gamma=gamma,
            lambda_=lambda_,
        ) # shape: B, L

        lambda_returns = lambda_returns.reshape(B*L)
        mask_padding = mask_padding.reshape(B*L)
        lambda_returns = lambda_returns[mask_padding]
        logits_actions = outputs.logits_actions.reshape(B*L, -1)[mask_padding]
        values = values.reshape(B*L)[mask_padding]
        actions = Categorical(logits=logits_actions).sample()

        d = Categorical(logits=logits_actions)
        log_probs = d.log_prob(actions)
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)


        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends,
                    loss_actions=loss_actions, loss_entropy=loss_entropy, loss_values=loss_values)

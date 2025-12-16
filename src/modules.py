import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import math


def info_nce_loss(query, key, temperature):
    """
    通用 InfoNCE 对比学习损失

    计算 query 和 key 之间的对比学习损失，正样本为对应位置的配对 (query[i], key[i])，
    负样本为 batch 内的其他所有 key[j] (j != i)

    Args:
        query: [B, d] - 查询向量（如预测的嵌入）
        key: [B, d] - 键向量（如目标嵌入），必须与 query 维度相同
        temperature: float 或 None - 温度参数，如果为 None 则使用 self.tau

    Returns:
        InfoNCE loss (标量)
    """
    assert query.shape == key.shape, \
        f"query and key must have the same shape, got {query.shape} vs {key.shape}"

    batch_size = query.size(0)

    # L2 归一化（对比学习的标准做法）
    query_norm = F.normalize(query, p=2, dim=1)  # [B, d]
    key_norm = F.normalize(key, p=2, dim=1)  # [B, d]

    # 计算正样本相似度（对角线元素：query[i] 与 key[i]）
    pos_sim = torch.sum(query_norm * key_norm, dim=1) / temperature  # [B]

    # 计算所有相似度矩阵（query[i] 与所有 key[j]）
    # [B, d] @ [d, B] = [B, B]
    all_sim = torch.matmul(query_norm, key_norm.T) / temperature  # [B, B]

    # InfoNCE loss: -log(exp(sim(q_i, k_i)) / sum_j(exp(sim(q_i, k_j))))
    # 使用 log-sum-exp trick 保证数值稳定性
    loss = -pos_sim + torch.logsumexp(all_sim, dim=1)

    return loss.mean()


class SinusoidalPositionalEmbedding(nn.Module):
    """
    用于将扩散步数 t 编码为向量
    """

    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 简单的 MLP 混合器，如果需要更复杂的 Attention 结构可以复用 TransformerEncoder


class DenoiseNetwork(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, out_dims)
        )

    def forward(self, x):
        return self.net(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0,
                                                      end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MLPDiffuser(nn.Module):
    """
    DreamRec 使用的 MLP 扩散网络
    输入: x_t (Noisy Item), step (t), condition (SASRec output)
    输出: Predicted Noise
    """

    def __init__(self, hidden_size, dims=None):
        super(MLPDiffuser, self).__init__()
        if dims is None:
            dims = hidden_size * 4

        self.time_emb = TimestepEmbedding(hidden_size)

        # 输入维度 = Item_Emb(H) + Condition(H) + Time_Emb(H)
        self.in_layers = nn.Sequential(
            nn.Linear(hidden_size * 3, dims),
            nn.SiLU(),
            nn.Linear(dims, dims),
            nn.SiLU(),
            nn.Linear(dims, hidden_size)
        )

    def forward(self, x, t, condition):
        # x: [B, H], t: [B], condition: [B, H]
        t_emb = self.time_emb(t)
        # Concatenate inputs
        emb = torch.cat([x, condition, t_emb], dim=-1)
        return self.in_layers(emb)


@staticmethod
def normalize_embed(x, dim=-1, eps=1e-8):
    """
    保留函数，当前不使用。
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


@staticmethod
def slerp(x0, x1, t, eps=1e-8):
    """
    保留函数，当前不使用。
    """
    if t.dim() == x0.dim() - 1:
        t = t.unsqueeze(-1)

    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small_angle = sin_theta.abs() < eps

    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    w0_lerp = 1 - t
    w1_lerp = t

    w0 = torch.where(small_angle, w0_lerp, w0)
    w1 = torch.where(small_angle, w1_lerp, w1)

    return w0 * x0 + w1 * x1


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()

        hidden_size = args.hidden_size
        inner_size = 4 * args.hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(args.hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi)
                       * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


#######################
## Basic Transformer ##
#######################

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.args = args
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(
            args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)  # TODO
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(
            mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(
            mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(
            mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super(TransformerBlock, self).__init__()
        self.layer = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        block = TransformerBlock(args)  # self attention

        self.blocks = nn.ModuleList([copy.deepcopy(block)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):

        all_encoder_layers = []

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        # 始终包含最后一层的输出
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers


class BidirectionalTransformer(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer, self).__init__()
        self.args = args

        # 1. Embeddings
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # 2. Encoder Body (Reusing your provided TransformerEncoder)
        # Ensure args.num_hidden_layers = 4, args.num_attention_heads = 4 in your config
        self.encoder = TransformerEncoder(args)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights. """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用可配置的初始化范围，默认为0.02（BERT标准）
            init_range = getattr(self.args, 'initializer_range', 0.02)
            module.weight.data.normal_(mean=0.0, std=init_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids):
        # input_ids: [Batch, Seq_Len]
        seq_length = input_ids.size(1)

        # --- 1. Generate Mask (Bidirectional, only mask padding) ---
        # Valid positions are 1, Padding is 0
        attention_mask = (input_ids > 0).long()  # [B, L]

        # Reshape for Multi-Head Attention broadcasting: [B, 1, 1, L]
        # 在计算 attention_scores [B, heads, L, L] 时会被广播
        # 每个 query 位置都会加上相同的 mask，只屏蔽 padding 的 key 位置
        # 这实现了真正的双向注意力（每个位置可以看到所有非padding位置）
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 转换为模型参数的dtype，支持混合精度训练（FP16/BF16）
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)

        # Convert to additive mask:
        # 1.0 (valid) -> 0.0 (加到 attention_scores 上不影响)
        # 0.0 (pad) -> -10000.0 (加到 attention_scores 上后 softmax 接近 0)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # --- 2. Embeddings ---
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        items_emb = self.item_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)

        embeddings = items_emb + position_emb
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # --- 3. Transformer Encoder ---
        # output_all_encoded_layers=False 时只返回最后一层的输出
        all_encoder_layers = self.encoder(
            embeddings, extended_attention_mask, output_all_encoded_layers=False)
        sequence_output = all_encoder_layers[-1]  # [Batch, Seq_Len, Hidden]

        return sequence_output

# class MMLPPredictor(nn.Module):
#     """
#     MLP-based Predictor with modulation, using adaLN-Zero approach to handle time and condition information.
#     Processes timestep embeddings and conditional encodings through modulated MLP blocks.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.hidden_size

#         # 时间嵌入MLP
#         self.time_mlp = TimeProjMLP(args)

#         # 条件投影（将x0映射到时间嵌入维度）
#         self.cond_proj = nn.Linear(args.hidden_size, args.hidden_size)

#         # 输入投影：将拼接后的 [xt, x0] 从 2*hidden_size 投影到 hidden_size
#         self.input_proj = nn.Linear(2 * args.hidden_size, args.hidden_size)

#         # 主干网络
#         self.blocks = nn.ModuleList([
#             ModulatedMLPBlock(args) for _ in range(args.predictor_blocks)
#         ])

#         # 最终输出层
#         self.final_norm = nn.LayerNorm(
#             args.hidden_size, elementwise_affine=False)
#         self.final_ada_lin = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(args.hidden_size, 2 * args.hidden_size))  # 生成scale和shift
#         self.head = nn.Linear(args.hidden_size, args.hidden_size)

#         # 初始化权重
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, xt, x0, t):
#         # 1. 将xt和x0连接成特征向量
#         combined_input = torch.cat([xt, x0], dim=-1)  # [B, 2 * hidden_size]

#         # 2. 获取时间嵌入
#         t_emb = self.time_mlp(t)  # [B, hidden_size]

#         # 3. 将x0投影为条件嵌入
#         cond_emb = self.cond_proj(x0)  # [B, hidden_size]

#         # 4. 构造全局条件向量 c = t_emb + cond_emb
#         c = t_emb + cond_emb  # [B, hidden_size]

#         # 5. 投影连接后的输入到隐藏维度
#         x = self.input_proj(combined_input)  # [B, hidden_size]

#         # 6. 通过多个调制MLP块
#         for block in self.blocks:
#             x = block(x, c)

#         # 7. 最终调制和输出
#         c_final = self.final_ada_lin(c)
#         scale, shift = c_final.chunk(2, dim=1)

#         x = self.final_norm(x)
#         x = x * (1 + scale) + shift
#         x = self.head(x)

#         return x


# class ModulatedMLPBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.hidden_size

#         # 归一化层（无affine参数，因为我们要动态生成）
#         self.norm = nn.LayerNorm(args.hidden_size, elementwise_affine=False)

#         # 前馈网络
#         self.ffn = nn.Sequential(
#             nn.Linear(args.hidden_size, args.hidden_size * 4),
#             nn.SiLU(),
#             nn.Linear(args.hidden_size * 4, args.hidden_size)
#         )

#         # 调制网络：根据条件c生成(scale, shift, gate)
#         self.ada_lin = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(args.hidden_size, 3 * args.hidden_size)
#         )

#         # 零初始化调制层的最后一个线性层
#         with torch.no_grad():
#             self.ada_lin[-1].weight.zero_()
#             self.ada_lin[-1].bias.zero_()

#         self.dropout = nn.Dropout(args.predictor_dropout_prob)

#     def forward(self, x, c):
#         # 1. 生成调制参数
#         modulation_params = self.ada_lin(c)
#         scale, shift, gate = modulation_params.chunk(3, dim=1)

#         # 2. 归一化
#         x_norm = self.norm(x)

#         # 3. 调制：归一化 -> 缩放 -> 平移
#         x_modulated = x_norm * (1 + scale) + shift

#         # 4. 应用前馈网络
#         h = self.ffn(x_modulated)
#         h = self.dropout(h)

#         # 5. 门控残差连接
#         return x + gate * h


# class GatedResNetPredictor(nn.Module):
#     """
#     A gated residual network predictor for flow matching models.

#     This predictor uses adaptive layer normalization (AdaLN) conditioned on time embeddings
#     to predict the velocity field in the flow matching framework. It takes the current state xt,
#     the initial condition x0, and time t as inputs, and outputs the predicted velocity.

#     The architecture consists of:
#     - Time embedding projection via TimeProjMLP
#     - Input projection to combine xt and x0
#     - Multiple ResBlocks with gated residual connections
#     - Final normalization and output head
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         self.time_mlp = TimeProjMLP(args)
#         self.input_proj = nn.Linear(args.hidden_size * 2, args.hidden_size)
#         self.blocks = nn.ModuleList([ResBlock(args)
#                                     for _ in range(args.predictor_blocks)])
#         self.final_norm = nn.LayerNorm(args.hidden_size)
#         self.head = nn.Linear(args.hidden_size, args.hidden_size)

#     def forward(self, xt, x0, t):
#         # t: [B] —— 已确保是 1D 向量
#         t_emb = self.time_mlp(t)  # [B, dim]

#         # 融合 Condition (Concat -> Linear)
#         x = torch.cat([xt, x0], dim=-1)  # [B, 2*dim]
#         x = self.input_proj(x)             # [B, dim]

#         # ResNet 处理
#         for block in self.blocks:
#             x = block(x, t_emb)

#         x = self.final_norm(x)
#         x = self.head(x)  # [B, dim]

#         return x


# class ResBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         self.proj_in = nn.Linear(args.hidden_size, args.hidden_size * 2)
#         self.act = nn.SiLU()
#         self.gate = nn.Linear(args.hidden_size, args.hidden_size * 2)
#         self.proj_out = nn.Linear(args.hidden_size * 2, args.hidden_size)
#         self.norm = nn.LayerNorm(args.hidden_size)
#         self.dropout = nn.Dropout(args.predictor_dropout_prob)

#     def forward(self, x, t_emb):
#         # x: [B, dim], t_emb: [B, dim]
#         shortcut = x
#         x = self.norm(x)
#         x = x + t_emb  # 注入时间信息

#         h = self.proj_in(x)
#         g = self.gate(x)
#         x = self.act(h) * torch.sigmoid(g)

#         x = self.proj_out(x)
#         x = self.dropout(x)
#         return x + shortcut


# class CrossAttnPredictor(nn.Module):
#     """
#     Cross-attention based predictor for flow matching.

#     This predictor uses cross-attention mechanism to predict the velocity field
#     by attending from the noisy state xt (query) to the clean state x0 (key/value),
#     conditioned on the time embedding.
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         self.time_mlp = TimeProjMLP(args)
#         self.xt_proj = nn.Linear(args.hidden_size, args.hidden_size)
#         self.x0_proj = nn.Linear(args.hidden_size, args.hidden_size)
#         self.blocks = nn.ModuleList([SimplifiedCrossAttnBlock(args)
#                                      for _ in range(args.predictor_blocks)])
#         self.head = nn.Linear(args.hidden_size, args.hidden_size)

#     def forward(self, xt, x0, t):
#         # t: [B] —— 确保是 1D 输入
#         t_emb = self.time_mlp(t)  # [B, dim]

#         # Query = xt + time embedding
#         query = self.xt_proj(xt) + t_emb  # [B, dim]
#         query = query.unsqueeze(1)        # [B, 1, dim]

#         # Key/Value = x0
#         kv = self.x0_proj(x0).unsqueeze(1)  # [B, 1, dim]

#         # Cross Attention
#         out = query
#         for block in self.blocks:
#             out = block(out, kv)  # [B, 1, dim]

#         out = out.squeeze(1)  # [B, dim]
#         out = self.head(out)

#         return out


# class SimplifiedCrossAttnBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             args.hidden_size, args.predictor_attention_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(args.hidden_size)
#         self.norm2 = nn.LayerNorm(args.hidden_size)
#         self.ffn = nn.Sequential(
#             nn.Linear(args.hidden_size, args.hidden_size * 4),
#             nn.SiLU(),
#             nn.Linear(args.hidden_size * 4, args.hidden_size)
#         )
#         self.attn_dropout = nn.Dropout(args.predictor_dropout_prob)
#         self.ffn_dropout = nn.Dropout(args.predictor_dropout_prob)

#     def forward(self, query, key_value):
#         # query: [B, 1, dim], key_value: [B, 1, dim]
#         attn_out, _ = self.attn(query, key_value, key_value)
#         attn_out = self.attn_dropout(attn_out)
#         x = self.norm1(query + attn_out)
#         ffn_out = self.ffn(x)
#         ffn_out = self.ffn_dropout(ffn_out)
#         x = self.norm2(x + ffn_out)
#         return x


# class MLPPredictor(nn.Module):
#     """
#     MLP-based predictor for flow matching.

#     输入: concat([xt, x0, t_emb])，其中 t_emb 是时间 t 的嵌入 (TimeProjMLP)。
#     输出: 预测的 x1_hat，形状 [B, hidden_size]
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.hidden_size

#         self.time_mlp = TimeProjMLP(args)

#         in_dim = args.hidden_size * 3  # xt, x0, t_emb
#         dropout = getattr(args, "predictor_dropout_prob", 0.0)

#         # 4 层 MLP（含输出层）：满足“3-4 层”要求
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, args.hidden_size * 2),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(args.hidden_size * 2, args.hidden_size * 2),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(args.hidden_size * 2, args.hidden_size),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(args.hidden_size, args.hidden_size),
#         )

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, xt, x0, t):
#         # t 允许是 [B] 或 [B,1]，TimeProjMLP 内部会 squeeze 成 [B]
#         t_emb = self.time_mlp(t)  # [B, dim]
#         x = torch.cat([xt, x0, t_emb], dim=1)  # [B, 3*dim]
#         return self.net(x)

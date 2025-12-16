import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.abstract import GenerativeSequentialRecommender
from modules import BidirectionalTransformer


class FlowSRModel(GenerativeSequentialRecommender):
    """
    Flow-based Sequential Recommender Model.

    This model uses Conditional Flow Matching (CFM) to learn the transformation
    from sequence representation (x0) to target item embedding (x1).
    """

    def __init__(self, args):
        super(FlowSRModel, self).__init__(args)
        self.args = args

        # Item encoder: bidirectional transformer for sequence encoding
        self.item_encoder = BidirectionalTransformer(args)
        self.item_embeddings = self.item_encoder.item_embeddings

        # Flow model: predictor network for learning the flow field
        self.flow_model = AdaLNPredictor(args)

        # Loss weights
        self.w_cfm = args.w_cfm  # CFM loss weight
        self.w_ce = args.w_ce  # Cross-entropy loss weight

        # Number of steps for inference
        self.n_steps = args.n_steps

    def get_seq_rep(self, input_ids):
        """
        Get sequence representation from input sequence.

        Args:
            input_ids: Input sequence of item IDs [batch_size, seq_len]

        Returns:
            x0: Sequence representation [batch_size, hidden_size]
        """
        seq_output = self.item_encoder(input_ids)
        # Take the last position as sequence representation
        x0 = seq_output[:, -1, :]
        return x0

    def forward(self, input_ids, target_ids):
        """
        Forward pass for training.

        Args:
            input_ids: Input sequence of item IDs [batch_size, seq_len]
            target_ids: Target item IDs [batch_size]

        Returns:
            x1_hat: Predicted target item embedding [batch_size, hidden_size]
            x0: Sequence representation [batch_size, hidden_size]
            x1: Ground truth target item embedding [batch_size, hidden_size]
        """
        # 1. Get sequence representation x0
        x0 = self.get_seq_rep(input_ids)

        # 2. Get target item embedding x1
        x1 = self.item_embeddings(target_ids)

        batch_size = x0.size(0)
        device = x0.device

        # 3. Sample time t uniformly from [0, 1]
        t = torch.rand(batch_size, device=device).unsqueeze(
            1)  # [batch_size, 1]

        # 4. Construct xt using linear interpolation: xt = (1-t)*x0 + t*x1
        xt = (1 - t) * x0 + t * x1

        # 5. Predict x1 using flow model
        # Note: t is [batch_size, 1], flow_model will handle the shape internally
        x1_hat = self.flow_model(xt, x0, t)

        return x1_hat, x0, x1

    def calculate_loss(self, input_ids, target_ids, neg_answers, same_target, user_ids):
        """
        Calculate the total loss for training.

        Args:
            input_ids: Input sequence of item IDs [batch_size, seq_len]
            target_ids: Target item IDs [batch_size]
            neg_answers: Negative samples (unused in this implementation)
            same_target: Whether same target (unused in this implementation)
            user_ids: User IDs (unused in this implementation)

        Returns:
            loss: Total weighted loss (CFM loss + CE loss)
        """
        x1_hat, x0, x1 = self.forward(input_ids, target_ids)

        # CFM loss: Mean squared error between predicted and ground truth x1
        # This loss comes from the flow model's prediction
        loss_cfm = F.mse_loss(x1_hat, x1)

        # Cross-entropy loss: Classification loss using sequence representation
        # This loss comes from the item encoder's representation
        all_items = self.item_embeddings.weight  # [item_size, hidden_size]
        # [batch_size, item_size]
        logits = torch.matmul(x0, all_items.transpose(0, 1))
        loss_ce = F.cross_entropy(logits, target_ids)

        # Weighted total loss
        weighted_loss_cfm = self.w_cfm * loss_cfm
        weighted_loss_ce = self.w_ce * loss_ce
        loss = weighted_loss_cfm + weighted_loss_ce

        return loss

    def predict_full(self, input_ids):
        """
        Full inference procedure using Euler integration.

        Starting from x0, iteratively apply the flow model to transform
        to x1 through N discrete steps.

        Args:
            input_ids: Input sequence of item IDs [batch_size, seq_len]

        Returns:
            scores: Item recommendation scores [batch_size, item_size]
        """
        # Initialize: start from sequence representation
        x0 = self.get_seq_rep(input_ids)
        xt = x0.clone()

        batch_size = xt.size(0)
        device = xt.device
        N = self.n_steps
        dt = 1.0 / N  # Time step size

        # Euler integration: iterate from t=0 to t=1
        for i in range(N):
            t_value = i / N  # Current time step in [0, 1]
            t = torch.full((batch_size,), t_value,
                           device=device)  # [batch_size]

            # Predict x1 at current time step
            x1_pred = self.flow_model(xt, x0, t)

            # Stop at the last step
            if i == N-1:
                break

            # Euler step: compute velocity and update xt
            # Velocity: vt = (x1_pred - xt) / (1 - t)
            vt = (x1_pred - xt) / (1 - t_value + 1e-8)
            # Update: xt = xt + vt * dt
            xt = xt + vt * dt

        # Final prediction
        x_hat = xt
        all_items = self.item_embeddings.weight
        # [batch_size, item_size]
        scores = torch.matmul(x_hat, all_items.transpose(0, 1))

        return scores


class AdaLNPredictor(nn.Module):
    """
    AdaLN-based predictor for flow matching.

    This model predicts the target embedding x1 from the interpolated state xt,
    conditioned on the sequence representation x0 and time step t.

    Architecture:
        - Time embedding via TimeProjMLP
        - Condition projection for x0
        - Multiple AdaLN blocks with adaptive normalization
        - Final adaptive layer normalization and output head

    Args:
        args: Configuration object containing:
            - hidden_size: Dimension of embeddings
            - predictor_blocks: Number of AdaLN blocks
            - predictor_dropout_prob: Dropout probability
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        # 1. 时间和条件的预处理
        self.time_mlp = TimeProjMLP(args)

        # 将 x0 映射到与时间嵌入相同的维度
        self.cond_proj = nn.Linear(args.hidden_size, args.hidden_size)

        # 2. 主干网络
        self.blocks = nn.ModuleList([
            AdaLNBlock(args) for _ in range(args.predictor_blocks)
        ])

        # 3. 最终输出层
        self.final_norm = nn.LayerNorm(
            args.hidden_size, elementwise_affine=False)
        self.final_ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.hidden_size, 2 * args.hidden_size)  # scale, shift
        )
        self.head = nn.Linear(args.hidden_size, args.hidden_size)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xt, x0, t):
        # --- 1. 构造全局 Condition 向量 c ---
        t_emb = self.time_mlp(t)       # [B, dim]
        cond_emb = self.cond_proj(x0)  # [B, dim]

        # 将时间和条件相加 (参考 DiT 设计，相加比拼接更紧凑且效果通常更好)
        c = t_emb + cond_emb           # [B, dim]

        # --- 2. 通过 AdaLN Blocks ---
        x = xt
        for block in self.blocks:
            x = block(x, c)

        # --- 3. 最终输出 ---
        # 最后一层也进行一次 AdaLN 处理
        c_final = self.final_ada_lin(c)
        scale, shift = c_final.chunk(2, dim=1)

        x = self.final_norm(x)
        x = x * (1 + scale) + shift
        x = self.head(x)

        return x


class AdaLNBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(args.predictor_dropout_prob)

        # 标准 LayerNorm，但在 AdaLN 中我们要手动预测 affine 参数，所以这里关掉自带的参数
        self.norm1 = nn.LayerNorm(args.hidden_size, elementwise_affine=False)

        # 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(args.hidden_size * 2, args.hidden_size)
        )

        # AdaLN 核心：根据条件 c 预测 (scale, shift, gate)
        # scale, shift 用于调节 Norm
        # gate 用于控制残差连接的强度 (Zero-init 策略)
        self.ada_lin = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.hidden_size, 3 * args.hidden_size)
        )

        # 对 gate 参数进行零初始化，这在扩散/流模型中对稳定性非常重要
        # 使得训练初始阶段 Block 近似恒等映射
        with torch.no_grad():
            self.ada_lin[-1].weight.zero_()
            self.ada_lin[-1].bias.zero_()

    def forward(self, x, c):
        # x: [B, dim]
        # c: [B, dim] (Condition vector)

        # 1. 生成调节参数
        chunks = self.ada_lin(c).chunk(3, dim=1)
        scale, shift, gate = chunks[0], chunks[1], chunks[2]

        # 2. Modulate (归一化 -> 缩放 -> 平移)
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale) + shift

        # 3. Apply FFN
        h = self.ffn(x_modulated)
        h = self.dropout(h)

        # 4. Gated Residual Connection
        # x_{l+1} = x_l + gate * FFN(AdaLN(x_l))
        return x + gate * h


class TimeProjMLP(nn.Module):
    """
    Time projection MLP that converts scalar timesteps into high-dimensional embeddings.

    Uses sinusoidal positional encoding followed by a two-layer MLP with SiLU activation.
    This allows the model to effectively condition on continuous time values in the flow matching process.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.SiLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )

    def get_sinusoidal_embeddings(self, timesteps, embedding_dim):
        assert len(
            timesteps.shape) == 1, f"Expected 1D timesteps, got {timesteps.shape}"
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t):
        # 明确要求 t 是 [B] 形状的一维张量
        if t.dim() == 2 and t.size(-1) == 1:
            t = t.squeeze(-1)  # 修复潜在广播错误 👈
        assert t.dim(
        ) == 1, f"Time tensor must be 1D after squeeze, got shape {t.shape}"

        t_emb = self.get_sinusoidal_embeddings(t, self.args.hidden_size)
        return self.net(t_emb)

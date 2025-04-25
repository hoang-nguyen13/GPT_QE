import math
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import holoviews as hv
import hvplot.pandas
import pandas as pd

# Configuration for GPT model
@dataclass
class GPTConfig:
    block_size: int = 4  # Sequence length for GQE
    vocab_size: int = None  # Set dynamically based on operator pool
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2
    bias: bool = False

# LayerNorm without bias option
class LayerNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Causal Self-Attention for transformer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device))[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# MLP for transformer block
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)

# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Base GPT Model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        print(f"num decayed params: {len(decay_params)}, {sum(p.numel() for p in decay_params):,}")
        print(f"num non-decayed params: {len(nodecay_params)}, {sum(p.numel() for p in nodecay_params):,}")
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

# GPT customized for GQE
class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        logits = self(current_tokens)
        next_token_mask = torch.nn.functional.one_hot(next_tokens, num_classes=self.config.vocab_size)
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        return torch.mean(torch.square(cumsum_logits - energies))

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 0] = float("inf")
            probs = F.softmax(-logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits

# Load H2 molecular data
def generate_molecule_data():
    datasets = qml.data.load("qchem", molname="H2")
    op_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)
    molecule_data = {}
    for dataset in datasets:
        molecule = dataset.molecule
        num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
        singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
        double_excs = [qml.DoubleExcitation(time, wires=double) for double in doubles for time in op_times]
        single_excs = [qml.SingleExcitation(time, wires=single) for single in singles for time in op_times]
        identity_ops = [qml.exp(qml.I(range(num_qubits)), 1j*time) for time in op_times]
        operator_pool = double_excs + single_excs + identity_ops
        molecule_data[dataset.molname] = {
            "op_pool": np.array(operator_pool),
            "num_qubits": num_qubits,
            "hf_state": dataset.hf_state,
            "hamiltonian": dataset.hamiltonian,
            "expected_ground_state_E": dataset.fci_energy
        }
    return molecule_data

# Define energy circuit
dev = qml.device("default.qubit", wires=4)  # H2 has 4 qubits
@qml.qnode(dev)
def energy_circuit(gqe_ops, init_state, hamiltonian):
    qml.BasisState(init_state, wires=range(4))
    for op in gqe_ops:
        qml.Snapshot(measurement=qml.expval(hamiltonian))
        qml.apply(op)
    return qml.expval(hamiltonian)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(op_seq, init_state, hamiltonian):
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops, init_state, hamiltonian)
        energies.append([es[k].item() for k in list(range(1, len(ops))) + ["execution_results"]])
    return np.array(energies)

# Main GQE training and evaluation
def main():
    # Load data
    molecule_data = generate_molecule_data()
    h2_data = molecule_data["H2"]
    op_pool = h2_data["op_pool"]
    num_qubits = h2_data["num_qubits"]
    init_state = h2_data["hf_state"]
    hamiltonian = h2_data["hamiltonian"]
    grd_E = h2_data["expected_ground_state_E"]
    op_pool_size = len(op_pool)

    # Generate dataset
    train_size = 1024
    seq_len = 4
    train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, seq_len))
    train_op_seq = op_pool[train_op_pool_inds]
    train_token_seq = np.concatenate([np.zeros(shape=(train_size, 1), dtype=int), train_op_pool_inds + 1], axis=1)
    train_sub_seq_en = get_subsequence_energies(train_op_seq, init_state, hamiltonian)

    # Initialize model
    config = GPTConfig(vocab_size=op_pool_size + 1, block_size=seq_len)
    gpt = GPTQE(config).to("cuda")
    opt = gpt.configure_optimizers(weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999))

    # Training loop
    n_batches = 8
    train_inds = np.arange(train_size)
    tokens = torch.from_numpy(train_token_seq).to("cuda")
    energies = torch.from_numpy(train_sub_seq_en).to("cuda")
    losses = []
    pred_Es_t = []
    true_Es_t = []
    current_mae = 10000
    gpt.train()

    for i in range(10000):
        np.random.shuffle(train_inds)
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        energy_batches = torch.tensor_split(energies[train_inds], n_batches)
        loss_record = 0
        for token_batch, energy_batch in zip(token_batches, energy_batches):
            opt.zero_grad()
            loss = gpt.calculate_loss(token_batch, energy_batch)
            loss.backward()
            opt.step()
            loss_record += loss.item() / n_batches
        losses.append(loss_record)

        if (i+1) % 500 == 0:
            gpt.eval()
            gen_token_seq, pred_Es = gpt.generate(n_sequences=100, max_new_tokens=seq_len, temperature=0.001, device="cuda")
            pred_Es = pred_Es.cpu().numpy()
            gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]
            true_Es = get_subsequence_energies(gen_op_seq, init_state, hamiltonian)[:, -1].reshape(-1, 1)
            mae = np.mean(np.abs(pred_Es - true_Es))
            ave_E = np.mean(true_Es)
            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)
            print(f"Iteration: {i+1}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}")
            if mae < current_mae:
                current_mae = mae
                torch.save(gpt, f"./seq_len={seq_len}_gqe.pt")
                print("Saved model!")
            gpt.train()

    pred_Es_t = np.concatenate(pred_Es_t, axis=1)
    true_Es_t = np.concatenate(true_Es_t, axis=1)

    # Visualization
    hvplot.extension('matplotlib')
    losses_df = pd.DataFrame(losses, columns=["0"])
    loss_fig = losses_df.hvplot(title="Training loss progress", ylabel="loss", xlabel="Training epochs", logy=True).opts(fig_size=600, fontscale=2, aspect=1.2)
    hv.render(loss_fig)

    df_true = pd.DataFrame(true_Es_t)
    df_pred = pd.DataFrame(pred_Es_t)
    df_true.columns = df_pred.columns = range(500, 10001, 500)
    df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
    df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]
    df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
    df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]
    fig = (
        df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    ) * (
        df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") *
        df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) *
        df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
    )
    fig = fig * hv.Curve([[0, grd_E], [10000, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
    fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
    hv.render(fig)

    # Compare sequence generation
    gen_token_seq, _ = gpt.generate(n_sequences=1024, max_new_tokens=seq_len, temperature=0.001, device="cuda")
    gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
    gen_op_seq = op_pool[gen_inds]
    true_Es = get_subsequence_energies(gen_op_seq, init_state, hamiltonian)[:, -1].reshape(-1, 1)
    loaded = torch.load(f"./seq_len={seq_len}_gqe.pt")
    loaded_token_seq, _ = loaded.generate(n_sequences=1024, max_new_tokens=seq_len, temperature=0.001, device="cuda")
    loaded_inds = (loaded_token_seq[:, 1:] - 1).cpu().numpy()
    loaded_op_seq = op_pool[loaded_inds]
    loaded_true_Es = get_subsequence_energies(loaded_op_seq, init_state, hamiltonian)[:, -1].reshape(-1, 1)
    df_compare_Es = pd.DataFrame({
        "Source": ["Random", "Latest Model", "Best Model"],
        "Aves": [train_sub_seq_en[:, -1].mean(), true_Es.mean(), loaded_true_Es.mean()],
        "Mins": [train_sub_seq_en[:, -1].min(), true_Es.min(), loaded_true_Es.min()],
        "Maxs": [train_sub_seq_en[:, -1].max(), true_Es.max(), loaded_true_Es.max()],
        "Mins_error": [
            abs(train_sub_seq_en[:, -1].min() - grd_E),
            abs(true_Es.min() - grd_E),
            abs(loaded_true_Es.min() - grd_E),
        ],
    })
    print(df_compare_Es)

if __name__ == "__main__":
    main()

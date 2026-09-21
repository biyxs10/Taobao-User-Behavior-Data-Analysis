import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import time

print("="*50)
print("步骤1: 读取建模数据")
save_dir = '/Users/neos/Desktop/Taobao DA/data_ready_for_modeling'
X_train = pd.read_parquet(f'{save_dir}/X_train.parquet')
X_val = pd.read_parquet(f'{save_dir}/X_val.parquet')
y_train = pd.read_parquet(f'{save_dir}/y_train.parquet')['is_buy']
y_val = pd.read_parquet(f'{save_dir}/y_val.parquet')['is_buy']
id_train = pd.read_parquet(f'{save_dir}/id_train.parquet')['user_id']
id_val = pd.read_parquet(f'{save_dir}/id_val.parquet')['user_id']
id_test = pd.read_parquet(f'{save_dir}/id_test.parquet')['user_id']
print(" 完成")

print("步骤2: 构建序列")
MAIN_PATH = '/Users/neos/Desktop/Taobao DA/user_behavior_cleaned'
df_all = pd.read_parquet(MAIN_PATH)
df_all['user_id'] = df_all['user_id'].astype('int64')
df_all['item_id'] = df_all['item_id'].astype('int64')
df_all['behavior_type'] = df_all['behavior_type'].astype('int8')
df_feature = df_all[df_all['time'] <= pd.Timestamp('2025-12-11 23:59:59')].copy()
seq_data = df_feature.sort_values(['user_id', 'time'])[['user_id', 'item_id', 'behavior_type']].copy()

N = 20
def build_sequence(group):
    items = group['item_id'].values[-N:].astype(np.int64)
    behaviors = group['behavior_type'].values[-N:].astype(np.int64)
    if len(items) < N:
        pad = N - len(items)
        items = np.concatenate([np.zeros(pad, dtype=np.int64), items])
        behaviors = np.concatenate([np.zeros(pad, dtype=np.int64), behaviors])
    return pd.Series({'item_seq': items, 'behavior_seq': behaviors})
#映射item——id
user_sequences = seq_data.groupby('user_id').apply(build_sequence).reset_index()
unique_items = seq_data['item_id'].unique()
item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
user_sequences['item_seq'] = user_sequences['item_seq'].apply(
    lambda seq: np.array([item_to_idx.get(i, 0) for i in seq])
)
print(f"  完成, item数: {len(unique_items)}")

print("步骤3: 切分序列")
item_seq_dict = dict(zip(user_sequences['user_id'], user_sequences['item_seq']))
behavior_seq_dict = dict(zip(user_sequences['user_id'], user_sequences['behavior_seq']))

def get_sequences(user_ids):
    return (np.array([item_seq_dict[uid] for uid in user_ids]),
            np.array([behavior_seq_dict[uid] for uid in user_ids]))

X_train_item, X_train_behavior = get_sequences(id_train.values)
X_val_item, X_val_behavior = get_sequences(id_val.values)
print("  完成")

# 转Tensor
X_train_item_t = torch.LongTensor(X_train_item)
X_train_behavior_t = torch.LongTensor(X_train_behavior)
y_train_t = torch.FloatTensor(y_train.values)
X_val_item_t = torch.LongTensor(X_val_item)
X_val_behavior_t = torch.LongTensor(X_val_behavior)
X_test_item, X_test_behavior = get_sequences(id_test.values)

train_dataset = TensorDataset(X_train_item_t, X_train_behavior_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

n_items = len(unique_items)
n_behaviors = 4
device = torch.device('cpu')

# ============================================================
print("="*50)
print("步骤4: 训练GRU模型（带早停）")
print("="*50)

class GRUModel(nn.Module):
    def __init__(self, n_items, n_behaviors, embed_dim=16, behavior_embed_dim=8, hidden_dim=32):
        super().__init__()
        self.item_embed = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.behavior_embed = nn.Embedding(n_behaviors + 1, behavior_embed_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=embed_dim + behavior_embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, item_seq, behavior_seq):
        item_emb = self.item_embed(item_seq)
        behavior_emb = self.behavior_embed(behavior_seq)
        x = torch.cat([item_emb, behavior_emb], dim=-1)
        _, hidden = self.gru(x)
        h = hidden[-1]
        return self.sigmoid(self.fc(h)).squeeze()

torch.manual_seed(42)
np.random.seed(42)

gru_model = GRUModel(n_items, n_behaviors).to(device)
print(f"  GRU参数量: {sum(p.numel() for p in gru_model.parameters()):,}")

criterion = nn.BCELoss()
optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

#早停
max_epochs = 30
patience = 3   #连续3轮不提升就停
best_auc = 0
best_state = None
no_improve = 0

print("  开始训练GRU...")
for epoch in range(max_epochs):
    gru_model.train()
    train_loss = 0
    for item_b, behavior_b, y_b in train_loader:
        optimizer.zero_grad()
        output = gru_model(item_b, behavior_b)
        loss = criterion(output, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    gru_model.eval()
    with torch.no_grad():
        val_pred = gru_model(X_val_item_t, X_val_behavior_t).numpy()
        val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"  GRU Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_auc={val_auc:.4f}")
    
    # 早停判断
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.clone() for k, v in gru_model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"  早停！最佳AUC={best_auc:.4f}（第{epoch+1-patience}轮）")
            break

#恢复最佳权重
gru_model.load_state_dict(best_state)
print(f"  GRU最终最佳验证AUC: {best_auc:.4f}")


# ============================================================
print("\n" + "="*50)
print("步骤5: 训练DIN模型（带早停）")
print("="*50)

class DIN_SelfAttention(nn.Module):
    def __init__(self, n_items, n_behaviors, embed_dim=16, behavior_embed_dim=8,
                 n_heads=2, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.item_embed = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.behavior_embed = nn.Embedding(n_behaviors + 1, behavior_embed_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(embed_dim + behavior_embed_dim, n_heads,
                                          batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim + behavior_embed_dim)
        self.fc = nn.Sequential(nn.Linear(embed_dim + behavior_embed_dim, hidden_dim),
                                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, item_seq, behavior_seq):
        item_emb = self.item_embed(item_seq)
        behavior_emb = self.behavior_embed(behavior_seq)
        x = torch.cat([item_emb, behavior_emb], dim=-1)
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x.mean(dim=1)
        return self.sigmoid(self.fc(x)).squeeze()

torch.manual_seed(42)
np.random.seed(42)

din_model = DIN_SelfAttention(n_items, n_behaviors).to(device)
print(f"  DIN参数量: {sum(p.numel() for p in din_model.parameters()):,}")

optimizer = optim.Adam(din_model.parameters(), lr=0.001)

best_auc_din = 0
best_state_din = None
no_improve_din = 0

print("  开始训练DIN...")
for epoch in range(max_epochs):
    din_model.train()
    train_loss = 0
    for item_b, behavior_b, y_b in train_loader:
        optimizer.zero_grad()
        output = din_model(item_b, behavior_b)
        loss = criterion(output, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    din_model.eval()
    with torch.no_grad():
        val_pred = din_model(X_val_item_t, X_val_behavior_t).numpy()
        val_auc = roc_auc_score(y_val, val_pred)
    
    print(f"  DIN Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_auc={val_auc:.4f}")
    
    if val_auc > best_auc_din:
        best_auc_din = val_auc
        best_state_din = {k: v.clone() for k, v in din_model.state_dict().items()}
        no_improve_din = 0
    else:
        no_improve_din += 1
        if no_improve_din >= patience:
            print(f"  早停！最佳AUC={best_auc_din:.4f}（第{epoch+1-patience}轮）")
            break

din_model.load_state_dict(best_state_din)
print(f" DIN最终最佳验证AUC: {best_auc_din:.4f}")

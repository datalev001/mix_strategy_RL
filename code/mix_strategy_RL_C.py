import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from gymnasium import Env, spaces
from tqdm.auto import trange
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 120

data_path = 'Per_Prom_mixRL.csv'
df = pd.read_csv(data_path)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Feature scaling
# ─────────────────────────────────────────────────────────────────────────────
feat_cols = [
    "age", "income_k", "region_code", "last_purchase_d",
    "browse_score", "device_type", "weekday", "is_holiday"
]

X = df[feat_cols].astype("float32").values
scaler = StandardScaler().fit(X)
Xn = scaler.transform(X)

feat_cols = ["age","income_k","region_code","last_purchase_d",
             "browse_score","device_type","weekday","is_holiday"]
X  = df[feat_cols].values.astype("float32")
scaler = StandardScaler().fit(X)
Xn = scaler.transform(X)

bgmm = BayesianGaussianMixture(
    n_components=10, weight_concentration_prior=0.1,
    covariance_type="diag", random_state=7
).fit(Xn)
df["cluster"] = bgmm.predict(Xn)
K = df.cluster.nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Gate Network ----------------
class Gate(nn.Module):
    def __init__(self,d,k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,64), nn.ReLU(),
            nn.Linear(64,k)
        )
    def forward(self,x):
        return torch.softmax(self.net(x), dim=-1)

gate     = Gate(len(feat_cols),K).to(device)
opt_gate = optim.AdamW(gate.parameters(), lr=1e-3)
ce       = nn.CrossEntropyLoss()

# ---------------- Bandit Env ----------------
class PromoEnv(Env):
    def __init__(self, slice_df):
        super().__init__()
        self.slice = slice_df.reset_index(drop=True)
        self.idx   = 0
        self.observation_space = spaces.Box(-5,5,(len(feat_cols),),np.float32)
        self.action_space      = spaces.Discrete(3)
    def _obs(self):
        return scaler.transform(self.slice.loc[[self.idx],feat_cols])[0]
    def reset(self, **kw):
        self.idx = rng.integers(len(self.slice))
        return self._obs(), {}
    def step(self, action):
        r = float(self.slice.loc[self.idx,"profit"])
        self.idx = rng.integers(len(self.slice))
        return self._obs(), r, True, False, {"profit":r}

# ---------------- Adaptive-Clustered PPO ----------------
ppo_heads = []
for k in range(K):
    sub = df[df.cluster==k]
    if len(sub)<3000:
        ppo_heads.append(None)
        continue
    env = DummyVecEnv([lambda sub=sub: Monitor(PromoEnv(sub))])
    m = PPO("MlpPolicy", env,
            n_steps=256, batch_size=128,
            learning_rate=3e-4, ent_coef=0.0,
            target_kl=0.15, device=device, verbose=0)
    m.set_logger(sb3_logger)
    ppo_heads.append(m)
    print(f"Cluster {k}: {len(sub):,} → PPO inited")

EPOCHS,BATCH = 80,2048
Xt   = torch.tensor(Xn, device=device)
acts = df.coupon.map({0:0,3:1,6:2}).values
prof = df.profit.values.astype("float32")

def get_mb(bs):
    idx = rng.integers(len(df), size=bs)
    return idx, Xt[idx], acts[idx], prof[idx]

hist = {"roi":[], "cvar":[]}
for ep in trange(1,EPOCHS+1, desc="Clustered PPO"):
    # fill buffers
    for _ in range(BATCH//256):
        idx,s,a,r = get_mb(256)
        with torch.no_grad(): gp = gate(s)
        chosen = gp.argmax(1).cpu().numpy()
        for k,m in enumerate(ppo_heads):
            if m is None: continue
            rows = np.where(chosen==k)[0]
            if not len(rows): continue
            buf = m.rollout_buffer
            for j in rows:
                if buf.full:
                    m.train(); buf.reset()
                v0 = m.policy.predict_values(s[j:j+1]).item()
                buf.add(obs=s[j].cpu().numpy(),
                        action=np.array([a[j]]),
                        reward=float(r[j]),
                        episode_start=np.array([True]),
                        value=torch.tensor([v0],device=device),
                        log_prob=torch.tensor([0.0],device=device))
    # every 5 epochs: fine-tune gate + offline eval
    if ep%5==0:
        idx,s_g,_,_ = get_mb(4000)
        with torch.no_grad():
            vals = [(m.policy.predict_values(s_g).squeeze(1) if m
                     else torch.zeros(len(s_g),device=device))
                    for m in ppo_heads]
        best = torch.stack(vals,1).argmax(1)
        loss = ce(gate(s_g), best)
        opt_gate.zero_grad(); loss.backward(); opt_gate.step()

        # offline eval
        vid = np.arange(20_000)
        s_val = Xt[vid]
        with torch.no_grad():
            ck = gate(s_val).argmax(1).cpu().numpy()
        rew, sp = [], []
        for i,ridx in enumerate(vid):
            k = ck[i]
            act,_ = (ppo_heads[k].predict(Xn[ridx], deterministic=True)
                     if ppo_heads[k] else (0,None))
            rew.append(df.profit.iat[ridx])
            sp.append([0,3,6][act])
        roi   = sum(rew)/(sum(sp)+1e-6)
        cvar  = np.percentile(rew,10)
        hist["roi"].append(roi); hist["cvar"].append(cvar)
        print(f"\nEpoch {ep:2d} | ROI={roi:.2f} | CVaR₀.₉={cvar:.2f}")

# ---------------- Single-head DQN ----------------
class SingleDQN(nn.Module):
    def __init__(self,obs, hid=64, act=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs,hid),nn.ReLU(),
            nn.Linear(hid,hid),nn.ReLU(),
            nn.Linear(hid,act))
    def forward(self,x): return self.net(x)

single = SingleDQN(len(feat_cols)).to(device)
opt_s  = optim.Adam(single.parameters(), lr=1e-3)
hubs   = nn.SmoothL1Loss()
for _ in range(10):
    idx = rng.integers(len(Xn), size=1024)
    sb  = torch.tensor(Xn[idx], device=device)
    ab  = torch.tensor(acts[idx], device=device)
    rb  = torch.tensor(prof[idx], device=device)
    qv  = single(sb).gather(1,ab.unsqueeze(1)).squeeze(1)
    loss= hubs(qv, rb)
    opt_s.zero_grad(); loss.backward(); opt_s.step()

# ---------------- Single-head PPO ----------------
env_full = DummyVecEnv([lambda: Monitor(PromoEnv(df))])
ppo_base = PPO("MlpPolicy", env_full,
               n_steps=256, batch_size=128,
               learning_rate=3e-4, ent_coef=0.0,
               target_kl=0.15, device=device, verbose=0)
ppo_base.set_logger(sb3_logger)
ppo_base.learn(total_timesteps=20_000)

# ---------------- Fixed-K DQN ----------------
OBS,ACT = len(feat_cols), 3
class EncFK(nn.Module):
    def __init__(self,in_d,hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d,hid),nn.ReLU(),
                                 nn.Linear(hid,hid),nn.ReLU())
    def forward(self,x): return self.net(x)
class HeadFK(nn.Module):
    def __init__(self,in_d,act):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(in_d,64),nn.ReLU(),
                               nn.Linear(64,act))
    def forward(self,h): return self.q(h)

enc_fk   = EncFK(OBS).to(device)
heads_fk = nn.ModuleList([HeadFK(64,ACT).to(device) for _ in range(K)])
opt_fk   = optim.Adam(list(enc_fk.parameters())+
                     [p for hd in heads_fk for p in hd.parameters()],
                     lr=1e-3)
huber_fk = nn.SmoothL1Loss(reduction="none")
X_t      = torch.tensor(Xn, device=device)
a_t      = torch.tensor(acts, device=device)
r_t      = torch.tensor(prof, device=device)

for _ in range(20):
    idx=rng.integers(0,len(df),size=1024)
    sb = X_t[idx]; ab = a_t[idx]; rb = r_t[idx]
    h  = enc_fk(sb)
    deltas = torch.stack([
        (rb - hd(h).gather(1,ab.unsqueeze(1)).squeeze(1)).detach()
        for hd in heads_fk],1)
    gamma = torch.softmax(-deltas.abs(),dim=1)
    loss_fk=0
    for k,hd in enumerate(heads_fk):
        qsa = hd(h).gather(1,ab.unsqueeze(1)).squeeze(1)
        loss_fk += (gamma[:,k]*huber_fk(qsa,rb)).mean()
    opt_fk.zero_grad(); loss_fk.backward(); opt_fk.step()

# ---------------- Compare first-10 & IPS ----------------
methods = ["Clustered PPO","DQN","PPO","Fixed-K DQN"]
ips     = {m:[] for m in methods}

first10 = {
    "Clustered PPO": [],
    "DQN": [],
    "PPO": [],
    "Fixed-K DQN": []
}

for i in range(10):
    o = Xn[i]
    # Clustered PPO
    kc = gate(torch.tensor(o, device=device).unsqueeze(0)).argmax(dim=1).item()
    if ppo_heads[kc] is not None:
        a_c, _ = ppo_heads[kc].predict(o, deterministic=True)
    else:
        a_c = 0
    a_c = int(a_c)

    # Single-head DQN
    qs = single(torch.tensor(o, device=device))
    a_d = int(qs.argmax().item())

    # Single-head PPO
    a_p, _ = ppo_base.predict(o, deterministic=True)
    a_p = int(a_p)

    # Fixed-K DQN
    h_fk  = enc_fk(torch.tensor(o, device=device).unsqueeze(0))
    qh_fk = torch.stack([hd(h_fk) for hd in heads_fk])
    best_k = qh_fk.max(2)[0].argmax(0).item()
    a_f   = int(qh_fk[best_k, 0].argmax().item())

    first10["Clustered PPO"].append(a_c)
    first10["DQN"].append(a_d)
    first10["PPO"].append(a_p)
    first10["Fixed-K DQN"].append(a_f)

for m in methods:
    print(f"First 10 actions, {m}: {first10[m]}")

p_log=1/3
for i in range(len(df)):
    true_a=acts[i]; r=prof[i]; o=Xn[i]
    # Clustered PPO
    kc = gate(torch.tensor(o,device=device).unsqueeze(0)).argmax(1).item()
    a_c,_ = (ppo_heads[kc].predict(o,deterministic=True) if ppo_heads[kc] else (0,None))
    ips["Clustered PPO"].append((a_c==true_a)*r/p_log)
    # DQN
    a_d = int(single(torch.tensor(o,device=device)).argmax().item())
    ips["DQN"].append((a_d==true_a)*r/p_log)
    # PPO
    a_p,_ = ppo_base.predict(o,deterministic=True)
    ips["PPO"].append((a_p==true_a)*r/p_log)
    # Fixed-K DQN
    h_fk  = enc_fk(torch.tensor(o,device=device).unsqueeze(0))
    qh_fk = torch.stack([hd(h_fk) for hd in heads_fk])
    bk    = qh_fk.max(2)[0].argmax(0).item()
    a_f   = int(qh_fk[bk,0].argmax().item())
    ips["Fixed-K DQN"].append((a_f==true_a)*r/p_log)

for m in methods:
    print(f"IPS profit — {m}: {np.mean(ips[m]):.2f}")

####
First 10 actions, Clustered PPO: [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
First 10 actions, DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
First 10 actions, PPO: [2, 2, 2, 2, 2, 0, 2, 2, 2, 2]
First 10 actions, Fixed-K DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

IPS profit — Clustered PPO: 3.11
IPS profit — DQN: 0.77
IPS profit — PPO: 1.22
IPS profit — Fixed-K DQN: 0.51

###
First 10 actions, Clustered PPO: [0, 0, 0, 0, 0, 1, 0, 0, 2, 0]
First 10 actions, DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
First 10 actions, PPO: [2, 2, 2, 1, 2, 1, 2, 2, 1, 2]
First 10 actions, Fixed-K DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
IPS profit — Clustered PPO: 2.28
IPS profit — DQN: 0.51
IPS profit — PPO: 4.92
IPS profit — Fixed-K DQN: 0.51
##############
First 10 actions, Clustered PPO: [2, 0, 1, 2, 2, 1, 2, 2, 2, 2]
First 10 actions, DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
First 10 actions, PPO: [0, 1, 0, 0, 0, 2, 0, 0, 0, 0]
First 10 actions, Fixed-K DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
IPS profit — Clustered PPO: 4.38
IPS profit — DQN: 3.08
IPS profit — PPO: 6.56
IPS profit — Fixed-K DQN: 0.51

###########
First 10 actions, Clustered PPO: [0, 1, 1, 0, 0, 2, 0, 1, 0, 0]
First 10 actions, DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
First 10 actions, PPO: [2, 0, 0, 0, 2, 0, 2, 0, 0, 2]
First 10 actions, Fixed-K DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
IPS profit — Clustered PPO: 3.95
IPS profit — DQN: 0.51
IPS profit — PPO: 7.27
IPS profit — Fixed-K DQN: 0.51
#########
First 10 actions, Clustered PPO: [1, 1, 1, 2, 0, 1, 1, 1, 1, 1]
First 10 actions, DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
First 10 actions, PPO: [2, 2, 0, 0, 2, 2, 2, 0, 2, 0]
First 10 actions, Fixed-K DQN: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
IPS profit — Clustered PPO: 1.54
IPS profit — DQN: 0.56
IPS profit — PPO: 2.26
IPS profit — Fixed-K DQN: 0.57


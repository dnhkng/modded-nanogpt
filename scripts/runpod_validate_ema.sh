#!/usr/bin/env bash
# Validate endgame EMA on 8xH100 -- RunPod native (pods are already containers,
# so no docker). Use a PyTorch template; needs 8x H100 SXM with NVLink.
#
#   ARMS=2 bash runpod_validate_ema.sh     # smoke: 1 baseline + 1 EMA
#   ARMS=8 bash runpod_validate_ema.sh     # full: 4 + 4
set -uo pipefail
REPO=${REPO:-https://github.com/dnhkng/modded-nanogpt.git}
BRANCH=${BRANCH:-endgame-ema-blend}
ARMS=${ARMS:-8}
EMA_HORIZON=${EMA_HORIZON:-150}
EMA_GAMMA=${EMA_GAMMA:-0.30}
CHUNKS=${CHUNKS:-9}
NPROC=${NPROC:-8}
WORK=${WORK:-/workspace/nanogpt-val}

echo "=== 0/5 topology check (MUST be NV#, not SYS/PHB) ==="
nvidia-smi topo -m | head -12
nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c
read -rp "Topology OK? [y/N] " ok; [ "$ok" = y ] || { echo "aborting"; exit 1; }

echo "=== 1/5 clone ==="
[ -d "$WORK" ] || git clone --quiet "$REPO" "$WORK"
cd "$WORK" && git fetch --quiet origin && git checkout --quiet "$BRANCH" \
  && git pull --quiet && git log --oneline -1

echo "=== 2/5 deps ==="
pip install -q -r requirements.txt

echo "=== 3/5 data (${CHUNKS}00M tokens) ==="
[ -f data/fineweb10B/fineweb_val_000000.bin ] || python data/cached_fineweb10B.py "$CHUNKS"
ls data/fineweb10B/*.bin | wc -l | xargs echo "  shards:"

run_one() {
    local tag=$1 use_ema=$2
    echo "  -> $tag  ($(date +%H:%M:%S))"
    if [ "$use_ema" = 1 ]; then
        EMA_HORIZON=$EMA_HORIZON EMA_GAMMA=$EMA_GAMMA \
          torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py > "logs/val_${tag}.txt" 2>&1
    else
        torchrun --standalone --nproc_per_node="$NPROC" train_gpt.py > "logs/val_${tag}.txt" 2>&1
    fi
    grep -oP 'step:\d+/\d+ val_loss:[\d.]+ train_time:\d+ms' "logs/val_${tag}.txt" | tail -1
}

echo "=== 4/5 runs (first includes ~7 min compile) ==="
mkdir -p logs
half=$((ARMS/2))
for i in $(seq 1 $half); do run_one "base_$i" 0; done
for i in $(seq 1 $half); do run_one "ema_$i"  1; done

echo "=== 5/5 results ==="
python3 - <<'PY'
import glob, re, statistics as st
def rows(pat):
    out=[]
    for f in sorted(glob.glob(pat)):
        pts=[(int(s),float(v),float(t)) for s,v,t in re.findall(
            r"step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms", open(f,errors="ignore").read())]
        if not pts: out.append((f,None,None)); continue
        hit=next((t for _,v,t in pts if v<=3.28), None)
        out.append((f, pts[-1][1], hit))
    return out
for label,pat in (("BASELINE","logs/val_base_*.txt"),("EMA","logs/val_ema_*.txt")):
    r=rows(pat); print(f"\n{label}")
    for f,fin,hit in r:
        print(f"  {f:<26} final={fin}  t_to_3.28={f'{hit/1000:.2f}s' if hit else 'NEVER'}")
    hs=[h for _,_,h in r if h]; fs=[x for _,x,_ in r if x]
    if hs: print(f"  mean t_to_3.28 {st.mean(hs)/1000:.2f}s"
                 + (f" (sd {st.stdev(hs)/1000:.2f})" if len(hs)>1 else "")
                 + f"   mean final {st.mean(fs):.4f}")
PY

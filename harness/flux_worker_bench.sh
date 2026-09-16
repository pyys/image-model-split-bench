#!/bin/bash
# FLUX worker(data) parallel bench - 1..4 independent workers, one per GPU
# build: /root/sdcpp-row (sd.cpp 68f3d6d)
# model: a FLUX.1-dev derived DiT, quantized to Q4_0 (see METHOD section 5-2)
#
# vae runs on each worker's OWN GPU (not CPU): with 4 workers a CPU VAE
# would serialize on 8 cores and measure CPU contention, not GPU throughput.
# te (T5) stays on CPU - 4.8GB does not fit alongside the DiT on one 8GB card.
#
# --batch-count 2 : image 1 absorbs kernel warm-up, image 2 is steady state.
#                   T5 conditioning runs ONCE per worker (verified by cond col).
set -u

SD=/root/sdcpp-row/build/bin/sd-cli
DIT=${DIT:-/root/models/flux-dit-q4_0.gguf}
VAE=/root/models/flux-sd-cpp/ae.safetensors
CLIP=/root/models/flux-sd-cpp/clip_l.safetensors
T5=/root/models/flux-sd-cpp/t5xxl-q8_0.gguf
OUT=/root/bench-flux/worker

PROMPT="a weathered brass compass on a wooden desk, morning light through a window, shallow depth of field"
W=512
H=512
STEPS=15
BATCH=4

mkdir -p "$OUT"
RES="$OUT/worker_results.tsv"
printf "workers\twall_s\tsamp_per_img\tsteady\tslowest\timg_per_min\tcond\ttemp_max\ttemp_avg\ttherm_pct\tpcap_pct\n" > "$RES"

# gpu csv fields: 1 timestamp, 2 index, 3 util, 4 mem, 5 temp, 6 power, 7 sm clock, 8 throttle
#
# clocks_throttle_reasons.active is a bitmask. 0x1 is GpuIdle, NOT throttling -
# an idle card sets it, so counting "non-zero" over-reports badly. Only these
# bits mean the clock was actually held down:
#   0x04 SwPowerCap   0x08 HwSlowdown   0x20 SwThermal   0x40 HwThermal
#   0x80 HwPowerBrake
gpu_summary () {
  awk -F, '
    function hexv(s,   i, c, v, n, h) {
      sub(/^0x/, "", s)
      n = length(s); if (n > 4) s = substr(s, n - 3, 4)
      h = "0123456789abcdef"; v = 0
      for (i = 1; i <= length(s); i++) {
        c = tolower(substr(s, i, 1)); v = v * 16 + (index(h, c) - 1)
      }
      return v
    }
    { gsub(/ /, "", $5); gsub(/ /, "", $8)
      t = $5 + 0
      if (t > mx) mx = t
      s += t; n++
      if (index($8, "0x") == 1) {
        v = hexv($8)
        if (int(v / 32) % 2 || int(v / 64) % 2 || int(v / 8) % 2) therm++
        if (int(v / 4) % 2 || int(v / 128) % 2) pcap++
      } }
    END { if (n > 0) printf "%d\t%.1f\t%.0f\t%.0f", mx, s / n, therm * 100 / n, pcap * 100 / n
          else printf "-\t-\t-\t-" }' "$1"
}

launch_worker () {
  n=$1
  gpu=$2
  log="$OUT/w${n}_gpu${gpu}.log"
  "$SD" --diffusion-model "$DIT" \
        --vae "$VAE" \
        --clip_l "$CLIP" \
        --t5xxl "$T5" \
        --backend "te=cpu,vae=cuda${gpu},diffusion=cuda${gpu}" \
        --diffusion-fa \
        --vae-tiling \
        -W $W -H $H \
        --steps $STEPS \
        --batch-count $BATCH \
        --cfg-scale 1.0 \
        --guidance 3.5 \
        --sampling-method euler \
        -s $((100 + gpu)) \
        -p "$PROMPT" \
        -o "${OUT}/n${n}_gpu${gpu}.png" \
        -v > "$log" 2>&1
}

run_set () {
  n=$1
  echo "=== ${n} worker(s)"

  gpulog="$OUT/gpu_w${n}.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active \
             --format=csv,noheader -l 1 > "$gpulog" 2>/dev/null &
  SMI=$!

  pids=""
  t0=$SECONDS
  g=0
  while [ $g -lt $n ]; do
    launch_worker "$n" "$g" &
    pids="$pids $!"
    g=$((g + 1))
  done
  for p in $pids; do
    wait $p
  done
  wall=$((SECONDS - t0))

  kill $SMI 2>/dev/null
  wait $SMI 2>/dev/null

  # per-image sampling seconds, per worker
  first=""
  second=""
  slow=0
  conds=0
  g=0
  while [ $g -lt $n ]; do
    wl="$OUT/w${n}_gpu${g}.log"
    a=$(grep -a "sampling completed" "$wl" | grep -oE "[0-9.]+s" | tr -d s | paste -sd/ -)
    b=$(grep -a "sampling completed" "$wl" | grep -oE "[0-9.]+s" | tr -d s | tail -1)
    c=$(grep -ac "get_learned_condition completed" "$wl")
    conds=$((conds + c))
    first="${first}${a:--} "
    second="${second}${b:--} "
    slow=$(awk -v x="$slow" -v y="${b:-0}" 'BEGIN{ print (y>x)?y:x }')
    g=$((g + 1))
  done

  ipm=$(awk -v n=$n -v s="$slow" 'BEGIN{ if (s>0) printf "%.2f", n*60/s; else print "-" }')

  gsum=$(gpu_summary "$gpulog")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
         "$n" "$wall" "$first" "$second" "$slow" "$ipm" "$conds" "$gsum" >> "$RES"
  echo "    wall=${wall}s  per-img=[${first}]  steady=[${second}]  slowest=${slow}s  -> ${ipm} img/min  cond=${conds}  temp/throttle=${gsum}"
  sleep 10
}

run_set 1
run_set 2
run_set 3
run_set 4

echo
echo "==== worker parallel results (512, 15 steps, Q4_0) ===="
column -t -s "$(printf '\t')" "$RES"
echo
echo "img1_s / img2_s are per-worker SAMPLING seconds (space separated, gpu0..gpuN-1)"
echo "img_per_min = workers x 60 / slowest img2      (steady-state GPU throughput)"
echo "logs: ${OUT}/wN_gpuG.log   gpu: ${OUT}/gpu_wN.csv   table: ${RES}"

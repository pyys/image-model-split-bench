#!/bin/bash
# FLUX layer/row split bench - 7 cells
# build: /root/sdcpp-row (sd.cpp 68f3d6d, ggml eced84c - split_buffer_type alive)
# model: a FLUX.1-dev derived DiT, quantized to Q4_0 (see METHOD section 5-2)
set -u

SD=/root/sdcpp-row/build/bin/sd-cli
DIT=${DIT:-/root/models/flux-dit-q4_0.gguf}
VAE=/root/models/flux-sd-cpp/ae.safetensors
CLIP=/root/models/flux-sd-cpp/clip_l.safetensors
T5=/root/models/flux-sd-cpp/t5xxl-q8_0.gguf
OUT=${OUT:-/root/bench-flux}

PROMPT="a weathered brass compass on a wooden desk, morning light through a window, shallow depth of field"
W=512
H=512
STEPS=15
SEED=42

mkdir -p "$OUT"
RES="$OUT/results.tsv"
printf "cell\tdevices\tmode\tstatus\twarm\tsampling\ttotal\tfallback\ttemp_max\ttemp_avg\ttherm_pct\tpcap_pct\n" > "$RES"

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

run_cell () {
  name=$1
  devs=$2
  mode=$3
  # ONLY=L2 runs just that cell. REPEAT=3 runs it 3 times (name gets _1 _2 _3).
  # Neither changes behaviour when unset.
  if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${name%%_*}" ]; then return; fi
  log="$OUT/r_${name}.log"
  gpulog="$OUT/gpu_${name}.csv"

  echo "=== ${name}  devices=${devs}  mode=${mode}"

  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active \
             --format=csv,noheader -l 1 > "$gpulog" 2>/dev/null &
  SMI=$!

  "$SD" --diffusion-model "$DIT" \
        --vae "$VAE" \
        --clip_l "$CLIP" \
        --t5xxl "$T5" \
        --backend "te=cpu,vae=cpu,diffusion=${devs}" \
        --split-mode "$mode" \
        --diffusion-fa \
        --vae-tiling \
        -W $W -H $H \
        --steps $STEPS \
        --cfg-scale 1.0 \
        --guidance 3.5 \
        --sampling-method euler \
        -s $SEED \
        -p "$PROMPT" \
        -o "${OUT}/img_${name}.png" \
        -v > "$log" 2>&1
  rc=$?

  kill $SMI 2>/dev/null
  wait $SMI 2>/dev/null

  warm=$(tr '\r' '\n' < "$log" | grep -oE "${STEPS}/${STEPS} - [0-9.]+ (s/it|it/s)" | tail -1)
  samp=$(grep -a "sampling completed" "$log" | grep -oE "[0-9.]+s" | tail -1)
  tot=$(grep -a "generate_image completed" "$log" | grep -oE "[0-9.]+s" | tail -1)
  fb=$(grep -ac "row split unavailable" "$log")

  st=OK
  if [ $rc -ne 0 ]; then st=FAIL; fi

  gsum=$(gpu_summary "$gpulog")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
         "$name" "$devs" "$mode" "$st" "${warm:--}" "${samp:--}" "${tot:--}" "$fb" "$gsum" >> "$RES"

  echo "    ${st}  warm=${warm:--}  sampling=${samp:--}  fallback=${fb}  temp/throttle=${gsum}"
  sleep 5
}

for i in $(seq 1 ${REPEAT:-1}); do
  if [ "${REPEAT:-1}" = 1 ]; then sfx=""; else sfx="_$i"; fi
  run_cell "1card$sfx" "cuda0"                   layer
  run_cell "L2$sfx"    "cuda0&cuda1"             layer
  run_cell "L3$sfx"    "cuda0&cuda1&cuda2"       layer
  run_cell "L4$sfx"    "cuda0&cuda1&cuda2&cuda3" layer
  run_cell "R2$sfx"    "cuda0&cuda1"             row
  run_cell "R3$sfx"    "cuda0&cuda1&cuda2"       row
  run_cell "R4$sfx"    "cuda0&cuda1&cuda2&cuda3" row
done

echo
echo "==== results ===="
column -t -s "$(printf '\t')" "$RES"
echo
echo "logs: ${OUT}/r_CELL.log   gpu: ${OUT}/gpu_CELL.csv   table: ${RES}"

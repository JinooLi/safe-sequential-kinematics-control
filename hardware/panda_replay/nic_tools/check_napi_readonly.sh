#!/usr/bin/env bash
# Bounded, temporary driver comparison. Only ICMP and libfranka state reads.
# No installed module, boot configuration, controller, hold, or recovery changes.
set -euo pipefail
[[ $EUID == 0 ]] || { echo 'Run with sudo.' >&2; exit 2; }
residual=0
trace_only=0
vendor=0
if [[ $# == 1 && $1 == --residual ]]; then residual=1
elif [[ $# == 1 && $1 == --trace-only ]]; then trace_only=1
elif [[ $# == 1 && $1 == --vendor ]]; then vendor=1
elif [[ $# != 0 ]]; then echo 'Options: --residual, --trace-only, or --vendor.' >&2; exit 2
fi
root=/home/cdsl/jinoo_ws/panda-hardware-replay
job="$root/nic_tools/r8169_napi_check_20261102"
probe="$root/network_isolation_20261102"
eth="$root/nic_tools/unpacked/sbin/ethtool"
stock=/lib/modules/5.9.1-rt20/kernel/drivers/net/ethernet/realtek/r8169.ko
module="$job/source/r8169.ko"
iface=enp4s0
operator_user=cdsl
expected_module_sha=0520ee206532228fa77aa30b5fc783985ded35901b201e9dd9db7a4d21a3fa18
candidate_name=r8169
if (( vendor )); then
  job="$root/nic_tools/r8125_vendor_20261102"
  module="$job/r8125-9.016.01/src/r8125.ko"
  expected_module_sha=8e04c1dda8436c2708dc9277d2e8ece00dc4bd894ecfeea180abdb9a2ed8432b
  candidate_name=r8125
  [[ ! -d /sys/module/r8125 ]] || { echo 'r8125 is already loaded; refusing.' >&2; exit 1; }
fi
[[ $(uname -r) == 5.9.1-rt20 ]]
[[ $(cat /sys/module/r8169/srcversion) == 4910F4E779F70AFA2802017 ]]
[[ $(sha256sum "$module" | cut -d ' ' -f1) == "$expected_module_sha" ]]
[[ $(/sbin/modinfo -F vermagic "$module") == $(/sbin/modinfo -F vermagic "$stock") ]]
exec 9>/run/lock/panda_napi_readonly.lock
flock -n 9 || { echo 'Another comparison is running.' >&2; exit 1; }
cd "$probe"
sha256sum --check probes.sha256
# A reload must affect only the robot NIC, never the SSH/network interface.
python3 - <<'PY'
from pathlib import Path
import json,subprocess
devices=[p.name for p in Path('/sys/bus/pci/drivers/r8169').glob('????:??:??.?')]
assert devices==['0000:04:00.0'],devices
addr=json.loads(subprocess.check_output(['ip','-j','address','show','dev','enp4s0']))[0]
assert addr['mtu']==1500 and 'UP' in addr['flags']
assert [(a['family'],a['local'],a['prefixlen']) for a in addr['addr_info']]==[('inet','172.16.0.1',24)]
routes=json.loads(subprocess.check_output(['ip','-j','route','show','dev','enp4s0']))
assert all(r['dst'] in ['172.16.0.0/24','169.254.0.0/16'] and 'gateway' not in r for r in routes),routes
PY
out=$(mktemp -d "$job/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
chown "$operator_user:$(id -gn "$operator_user")" "$out"
echo "Diagnostic directory: $out"
state_check() {
  local ready=0
  for attempt in {1..3}; do
    if runuser -u "$operator_user" -- timeout 6 "$probe/latency_probe" --state 172.16.0.2 > "$1" 2> "$1.stderr"; then
      ready=1
      break
    fi
    sleep .5
  done
  if (( ! ready )); then cat "$1.stderr" >&2; return 1; fi
  python3 - "$1" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['robot_mode'] in ['Idle','Reflex'],s['robot_mode']
assert max(map(abs,s['dq']))<0.01,'Robot is moving; refusing NIC changes.'
print('Robot mode:',s['robot_mode'],'State reception only.')
PY
}
state_check "$out/before_robot.json"
find_irq() {
  irq=$(awk '/enp4s0/{gsub(":","",$1);print $1}' /proc/interrupts)
  [[ $irq =~ ^[0-9]+$ ]]
  # Manufacturer IRQ names have a queue suffix; comm may be truncated to 15 chars.
  tid=$(ps -eLo tid,comm | awk -v n="irq/$irq-" 'index($2,n)==1{print $1}')
  [[ $tid =~ ^[0-9]+$ ]]
}
find_irq
original_affinity=$(cat "/proc/irq/$irq/smp_affinity_list")
original_priority=$(ps -o rtprio= -p "$tid" | tr -d ' ')
[[ $original_priority =~ ^[0-9]+$ ]]
original_gro_timeout=$(cat "/sys/class/net/$iface/gro_flush_timeout")
original_defer=$(cat "/sys/class/net/$iface/napi_defer_hard_irqs")
original_power=$(cat "/sys/class/net/$iface/device/power/control")
"$eth" -k "$iface" > "$out/entry_features.txt"
# Restore the previous failed job's GRO change before making a new comparison.
# No other entry-state differences are accepted.
feature_reference="$root/nic_tools/r8169_napi_check_20261102/run_20261102T053545Z_hUdcbE/before_features.txt"
[[ $(sha256sum "$feature_reference" | cut -d ' ' -f1) == 198f1e2c4253aed5f972e9b607d0f8ae5d9ddeb994f86b827d3ff682fd035d4c ]]
python3 - "$out/entry_features.txt" "$feature_reference" <<'PY'
from pathlib import Path
import sys
current=Path(sys.argv[1]).read_text()
reference=Path(sys.argv[2]).read_text()
assert current.replace('generic-receive-offload: on\n','generic-receive-offload: off\n')==reference,'Unexpected feature changes; refusing.'
PY
cp "$feature_reference" "$out/before_features.txt"
"$eth" --show-eee "$iface" > "$out/before_eee.txt"
grep -q 'EEE status: disabled' "$out/before_eee.txt"
grep -q '^generic-receive-offload: off' "$out/before_features.txt"
ip -j route show dev "$iface" > "$out/before_routes.json"
irqbalance_was_active=0
if systemctl is-active --quiet irqbalance; then irqbalance_was_active=1; fi
printf '%s\n' "affinity=$original_affinity priority=$original_priority gro=$original_gro_timeout defer=$original_defer power=$original_power irqbalance=$irqbalance_was_active" > "$out/original_parameters.txt"
driver_touched=0
capture_pid=
instance=
stop_capture() {
  if [[ -n $capture_pid ]]; then
    kill -INT "$capture_pid" 2>/dev/null || true
    wait "$capture_pid" || true
    capture_pid=
  fi
}
stop_trace() {
  if [[ -n $instance && -d $instance ]]; then
    printf '0\n' > "$instance/tracing_on" || return 1
    printf '0\n' > "$instance/events/enable" || return 1
    rmdir "$instance" || return 1
    instance=
  fi
}
configure_link() {
  local ready=0 failed=0
  for attempt in {1..50}; do
    if [[ -d /sys/class/net/$iface ]]; then ready=1; break; fi
    sleep .1
  done
  (( ready )) || return 1
  ip link set dev "$iface" up || failed=1
  ip address replace 172.16.0.1/24 dev "$iface" || failed=1
  python3 - "$out/before_routes.json" <<'PY' || failed=1
import json,subprocess,sys
for r in json.load(open(sys.argv[1])):
    if r['dst']=='169.254.0.0/16':
        subprocess.run(['ip','route','replace',r['dst'],'dev','enp4s0','scope','link','metric',str(r.get('metric',1000))],check=True)
PY
  # Use the short option names supported by this server's ethtool 5.4.
  # All other features match the stock defaults, verified below.
  "$eth" --set-eee "$iface" eee off || failed=1
  if [[ $(basename "$(readlink -f "/sys/class/net/$iface/device/driver")") == r8125 ]]; then
    "$eth" -K "$iface" rx on tx on sg off tso off gso off gro off || failed=1
  else
    "$eth" -K "$iface" gro off || failed=1
  fi
  printf '%s\n' "$original_power" > "/sys/class/net/$iface/device/power/control" || failed=1
  ready=0
  for attempt in {1..100}; do
    if [[ $(cat "/sys/class/net/$iface/carrier") == 1 ]]; then ready=1; break; fi
    sleep .1
  done
  (( ready )) || failed=1
  "$eth" -k "$iface" > "$out/current_features.txt" || failed=1
  if [[ $(basename "$(readlink -f "/sys/class/net/$iface/device/driver")") == r8125 ]]; then
    # Different drivers advertise different fixed capabilities. Check the controls
    # relevant to this test; compare the complete original feature set on restore.
    grep -q '^generic-receive-offload: off' "$out/current_features.txt" || failed=1
    grep -q '^rx-checksumming: on' "$out/current_features.txt" || failed=1
    grep -q '^tx-checksumming: on' "$out/current_features.txt" || failed=1
  else
    diff -u "$out/before_features.txt" "$out/current_features.txt" > "$out/current_features_diff.txt" || failed=1
  fi
  (( failed == 0 ))
}
restore_parameters() {
  local failed=0
  find_irq || return 1
  printf '%s\n' "$original_affinity" > "/proc/irq/$irq/smp_affinity_list" || failed=1
  chrt -f -p "$original_priority" "$tid" || failed=1
  printf '%s\n' "$original_gro_timeout" > "/sys/class/net/$iface/gro_flush_timeout" || failed=1
  printf '%s\n' "$original_defer" > "/sys/class/net/$iface/napi_defer_hard_irqs" || failed=1
  (( failed == 0 ))
}
load_stock() {
  local failed=0
  if (( vendor )) && [[ -d /sys/module/r8125 ]]; then /sbin/rmmod r8125 || return 1; fi
  if [[ -d /sys/module/r8169 ]]; then /sbin/rmmod r8169 || return 1; fi
  /sbin/insmod "$stock" || return 1
  configure_link || failed=1
  restore_parameters || failed=1
  [[ $(cat /sys/module/r8169/srcversion) == 4910F4E779F70AFA2802017 ]] || return 1
  driver_touched=0
  (( failed == 0 ))
}
cleanup() {
  local rc=$? failed=0
  trap - EXIT INT TERM
  set +e
  stop_capture
  stop_trace || failed=1
  if (( driver_touched )); then load_stock || failed=1; fi
  "$eth" -K "$iface" gro off || failed=1
  restore_parameters || failed=1
  if (( irqbalance_was_active )); then systemctl start irqbalance || failed=1; fi
  "$eth" -k "$iface" > "$out/after_features.txt" || failed=1
  diff -u "$out/before_features.txt" "$out/after_features.txt" > "$out/features_diff.txt" || failed=1
  "$eth" --show-eee "$iface" > "$out/after_eee.txt" || failed=1
  cat "/sys/class/net/$iface/gro_flush_timeout" "/sys/class/net/$iface/napi_defer_hard_irqs" > "$out/after_napi.txt"
  cat "/sys/class/net/$iface/device/power/control" > "$out/after_power.txt"
  ip -j address show dev "$iface" > "$out/after_address.json"
  ip -j route show dev "$iface" > "$out/after_routes.json"
  systemctl is-active irqbalance > "$out/after_irqbalance.txt"
  cat /sys/module/r8169/srcversion > "$out/after_srcversion.txt"
  cat "/proc/irq/$irq/smp_affinity_list" > "$out/after_irq_affinity.txt"
  chrt -p "$tid" > "$out/after_irq_priority.txt"
  state_check "$out/after_robot.json" || failed=1
  chown -R "$operator_user:$(id -gn "$operator_user")" "$out"
  if (( failed )); then echo "RESTORATION/VERIFICATION FAILED: $out" >&2; exit 1; fi
  echo "Stock driver and original parameters restored. Diagnostic directory: $out"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Correct the single remaining difference from the interrupted previous run.
"$eth" -K "$iface" gro off
"$eth" -k "$iface" > "$out/repaired_features.txt"
diff -u "$out/before_features.txt" "$out/repaired_features.txt"
if (( irqbalance_was_active )); then systemctl stop irqbalance; fi
run_case() {
  local label=$1 seconds=$2
  echo "START $label: ICMP and robot state reads; no tracing or control."
  local driver
  driver=$(basename "$(readlink -f "/sys/class/net/$iface/device/driver")")
  cat "/sys/module/$driver/srcversion" > "$out/${label}_srcversion.txt"
  "$eth" -i "$iface" > "$out/${label}_driver.txt"
  "$eth" -k "$iface" > "$out/${label}_features.txt"
  cat "/sys/class/net/$iface/gro_flush_timeout" "/sys/class/net/$iface/napi_defer_hard_irqs" > "$out/${label}_napi.txt"
  "$eth" -S "$iface" > "$out/${label}_stats_before.txt"
  runuser -u "$operator_user" -- timeout "$((seconds+4))" "$probe/icmp_probe" 172.16.0.2 2341 "$((seconds*1000))" "$out/$label" > "$out/${label}_icmp.log" 2>&1
  runuser -u "$operator_user" -- timeout "$((seconds+4))" "$probe/latency_probe" --read 3 "$seconds" "$out/$label" 172.16.0.2 > "$out/${label}_read.log" 2>&1
  "$eth" -S "$iface" > "$out/${label}_stats_after.txt"
  cat "$out/${label}_icmp.log" "$out/${label}_read.log"
}
trace_residual() {
  local trace=/sys/kernel/tracing label=continuous_rx_traced
  [[ -d $trace/instances ]] || trace=/sys/kernel/debug/tracing
  [[ -d $trace/instances ]] || return 1
  instance="$trace/instances/panda_residual_$$"
  mkdir "$instance"
  printf '0\n' > "$instance/tracing_on"
  printf 'mono\n' > "$instance/trace_clock"
  printf '32768\n' > "$instance/buffer_size_kb"
  # A running, pinned NAPI timer need not move with the IRQ affinity.
  printf '200008\n' > "$instance/tracing_cpumask" # CPUs 3 and 21.
  local event
  for event in irq/irq_handler_entry irq/irq_handler_exit; do
    printf 'irq == %s\n' "$irq" > "$instance/events/$event/filter"
    printf '1\n' > "$instance/events/$event/enable"
  done
  for event in napi/napi_poll net/napi_gro_receive_entry net/net_dev_start_xmit; do
    if [[ $event == napi/* ]]; then
      printf 'dev_name == "enp4s0"\n' > "$instance/events/$event/filter"
    else
      printf 'name == "enp4s0"\n' > "$instance/events/$event/filter"
    fi
    printf '1\n' > "$instance/events/$event/enable"
  done
  for event in sched/sched_switch timer/hrtimer_expire_entry irq/softirq_entry irq/softirq_exit; do
    if [[ -f $instance/events/$event/enable ]]; then
      printf '1\n' > "$instance/events/$event/enable"
    fi
  done
  echo 'START traced residual check: 8 seconds of robot state reception only.'
  taskset -c 0 /usr/sbin/tcpdump -Z "$operator_user" -i "$iface" -nn -s 0 -B 4096 \
    -w "$out/$label.pcap" 'host 172.16.0.2 and (udp or icmp)' > "$out/${label}_capture.log" 2>&1 &
  capture_pid=$!
  sleep .2
  kill -0 "$capture_pid"
  python3 -c 'import time; print(time.monotonic_ns(), time.time_ns())' > "$out/${label}_start_clock.txt"
  printf '1\n' > "$instance/tracing_on"
  runuser -u "$operator_user" -- timeout 12 "$probe/latency_probe" --read 3 8 "$out/$label" 172.16.0.2 > "$out/${label}_read.log" 2>&1
  printf '0\n' > "$instance/tracing_on"
  stop_capture
  # tracefs reports a synthetic file size; stream it to gzip through stdin.
  cat "$instance/trace" | gzip -c > "$out/${label}_trace.txt.gz"
  stop_trace
  cat "$out/${label}_read.log"
}
if (( ! trace_only )); then run_case stock_before 5; fi
state_check "$out/before_switch_robot.json"
driver_touched=1
/sbin/rmmod r8169
if (( vendor )); then
  /sbin/insmod "$module" aspm=0 eee_enable=0 eee_giga_lite=0 timer_count=0
else
  /sbin/insmod "$module"
fi
configure_link
restore_parameters
[[ $(cat "/sys/module/$candidate_name/srcversion") == $(/sbin/modinfo -F srcversion "$module") ]]
# Establish neighbor resolution and a working state connection before timing.
state_check "$out/after_switch_robot.json"
if (( vendor )); then
  run_case vendor_interrupts 20
  printf '100000\n' > "/sys/class/net/$iface/gro_flush_timeout"
  printf '1000\n' > "/sys/class/net/$iface/napi_defer_hard_irqs"
  run_case vendor_polled 20
elif (( trace_only )); then
  printf '100000\n' > "/sys/class/net/$iface/gro_flush_timeout"
  printf '1000\n' > "/sys/class/net/$iface/napi_defer_hard_irqs"
  trace_residual
elif (( residual )); then
  # Continue polling through short receive gaps instead of stopping after 2 ms.
  printf '100000\n' > "/sys/class/net/$iface/gro_flush_timeout"
  printf '1000\n' > "/sys/class/net/$iface/napi_defer_hard_irqs"
  run_case continuous_cpu21 20
  # Finish the pinned timer cycle before moving the interrupt and restarting it.
  printf '0\n' > "/sys/class/net/$iface/napi_defer_hard_irqs"
  printf '0\n' > "/sys/class/net/$iface/gro_flush_timeout"
  sleep .2
  printf '3\n' > "/proc/irq/$irq/smp_affinity_list"
  chrt -f -p 80 "$tid"
  printf '100000\n' > "/sys/class/net/$iface/gro_flush_timeout"
  printf '1000\n' > "/sys/class/net/$iface/napi_defer_hard_irqs"
  run_case continuous_cpu3 20
  trace_residual
else
  run_case patched_no_timer 5
  for setting in '20000 2' '100000 2' '100000 20'; do
    read -r interval defer <<< "$setting"
    printf '%s\n' "$interval" > "/sys/class/net/$iface/gro_flush_timeout"
    printf '%s\n' "$defer" > "/sys/class/net/$iface/napi_defer_hard_irqs"
    run_case "patched_${interval}ns_${defer}polls" 10
  done
fi
state_check "$out/before_restore_robot.json"
load_stock
state_check "$out/after_restore_robot.json"
if (( ! trace_only )); then run_case stock_after 5; fi
echo 'Comparison completed. The exit handler verifies restoration.'

#!/usr/bin/env bash
# Fixed, bounded diagnostic. No robot control, recovery, driver reload, or boot edits.
# NIC settings and irqbalance are restored on normal exit and handled signals.
set -euo pipefail
[[ $EUID == 0 && $# == 0 ]] || { echo 'Run this fixed diagnostic with sudo and no arguments.' >&2; exit 2; }
root=/home/cdsl/jinoo_ws/panda-hardware-replay
probe_dir="$root/network_isolation_20261102"
eth="$root/nic_tools/unpacked/sbin/ethtool"
iface=enp4s0
operator_user=cdsl
[[ $(uname -r) == 5.9.1-rt20 ]] || { echo 'Unexpected kernel.' >&2; exit 1; }
[[ $(cat /sys/module/r8169/srcversion) == 4910F4E779F70AFA2802017 ]] || { echo 'Expected stock NIC driver.' >&2; exit 1; }
cd "$probe_dir"
sha256sum --check probes.sha256
out=$(mktemp -d "$probe_dir/privileged_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
chown "$operator_user:$(id -gn "$operator_user")" "$out"
echo "Diagnostic directory: $out"
runuser -u "$operator_user" -- timeout 5 ./latency_probe --state 172.16.0.2 > "$out/before_robot.json"
python3 - "$out/before_robot.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s['robot_mode'] not in ['Idle','Reflex']:
    raise SystemExit('Robot is not Idle or Reflex; refusing diagnostics.')
print('Robot mode:',s['robot_mode'],'No control or recovery will be started.')
PY
irq=$(awk '/enp4s0/{gsub(":","",$1);print $1}' /proc/interrupts)
[[ $irq =~ ^[0-9]+$ ]] || { echo 'Expected one NIC IRQ.' >&2; exit 1; }
irq_tid=$(ps -eLo tid,comm | awk -v n="irq/$irq-$iface" '$2==n{print $1}')
[[ $irq_tid =~ ^[0-9]+$ ]] || { echo 'NIC IRQ thread not found.' >&2; exit 1; }
original_affinity=$(cat "/proc/irq/$irq/smp_affinity_list")
original_priority=$(ps -o rtprio= -p "$irq_tid" | tr -d ' ')
[[ $original_priority =~ ^[0-9]+$ ]] || { echo 'Expected FIFO IRQ thread.' >&2; exit 1; }
original_gro_timeout=$(cat "/sys/class/net/$iface/gro_flush_timeout")
original_defer=$(cat "/sys/class/net/$iface/napi_defer_hard_irqs")
"$eth" -k "$iface" > "$out/before_features.txt"
original_rx=$(awk '$1=="rx-checksumming:"{print $2}' "$out/before_features.txt")
original_tx=$(awk '$1=="tx-checksumming:"{print $2}' "$out/before_features.txt")
[[ $original_rx =~ ^(on|off)$ && $original_tx =~ ^(on|off)$ ]]
irqbalance_was_active=0
if systemctl is-active --quiet irqbalance; then irqbalance_was_active=1; fi
capture_pid=
instance=
restoration_failed=0
restore_parameters() {
  printf '%s\n' "$original_defer" > "/sys/class/net/$iface/napi_defer_hard_irqs" || return 1
  printf '%s\n' "$original_gro_timeout" > "/sys/class/net/$iface/gro_flush_timeout" || return 1
  "$eth" -K "$iface" rx "$original_rx" tx "$original_tx" || return 1
  printf '%s\n' "$original_affinity" > "/proc/irq/$irq/smp_affinity_list" || return 1
  chrt -f -p "$original_priority" "$irq_tid" || return 1
}
stop_capture() {
  if [[ -n $capture_pid ]]; then
    kill -INT "$capture_pid" 2>/dev/null || true
    wait "$capture_pid" || true
    capture_pid=
  fi
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  stop_capture
  if [[ -n $instance && -d $instance ]]; then
    printf '0\n' > "$instance/tracing_on"
    printf '0\n' > "$instance/events/enable"
    rmdir "$instance"
  fi
  if ! restore_parameters; then restoration_failed=1; fi
  if (( irqbalance_was_active )); then systemctl start irqbalance || restoration_failed=1; fi
  "$eth" -k "$iface" > "$out/after_features.txt"
  cat "/proc/irq/$irq/smp_affinity_list" > "$out/after_irq_affinity.txt"
  chrt -p "$irq_tid" > "$out/after_irq_priority.txt"
  cat "/sys/class/net/$iface/gro_flush_timeout" > "$out/after_gro_timeout.txt"
  cat "/sys/class/net/$iface/napi_defer_hard_irqs" > "$out/after_defer_irqs.txt"
  runuser -u "$operator_user" -- timeout 5 "$probe_dir/latency_probe" --state 172.16.0.2 > "$out/after_robot.json"
  chown -R "$operator_user:$(id -gn "$operator_user")" "$out"
  if (( restoration_failed )); then
    echo "RESTORATION FAILED; inspect $out before continuing." >&2
    exit 1
  fi
  echo "Original NIC parameters restored. Diagnostic directory: $out"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if (( irqbalance_was_active )); then systemctl stop irqbalance; fi
printf '%s\n' "irq=$irq tid=$irq_tid affinity=$original_affinity priority=$original_priority rx=$original_rx tx=$original_tx gro_timeout=$original_gro_timeout defer=$original_defer irqbalance=$irqbalance_was_active" > "$out/original_parameters.txt"
trace=/sys/kernel/tracing
[[ -d $trace/instances ]] || trace=/sys/kernel/debug/tracing
[[ -d $trace/instances ]] || { echo 'Independent ftrace instances unavailable.' >&2; exit 1; }
instance="$trace/instances/panda_network_$$"
mkdir "$instance"
printf '0\n' > "$instance/tracing_on"
printf 'mono\n' > "$instance/trace_clock"
printf '8192\n' > "$instance/buffer_size_kb"
printf '200008\n' > "$instance/tracing_cpumask" # CPU 3 and CPU 21 only.
enable_event() {
  local event=$1 expression=$2
  [[ -f $instance/events/$event/enable ]] || { echo "Unavailable event: $event"; return; }
  printf '%s\n' "$expression" > "$instance/events/$event/filter"
  printf '1\n' > "$instance/events/$event/enable"
}
enable_event irq/irq_handler_entry "irq == $irq"
enable_event irq/irq_handler_exit "irq == $irq"
enable_event irq/softirq_entry 'vec == 2 || vec == 3'
enable_event irq/softirq_exit 'vec == 2 || vec == 3'
enable_event napi/napi_poll 'dev_name == "enp4s0"'
enable_event net/net_dev_start_xmit 'name == "enp4s0"'
enable_event net/napi_gro_receive_entry 'name == "enp4s0"'
enable_event sched/sched_waking "pid == $irq_tid"
enable_event sched/sched_switch "prev_pid == $irq_tid || next_pid == $irq_tid"
"$eth" -d "$iface" > "$out/register_dump.txt" 2>&1 || true
lspci -s 04:00.0 -vv > "$out/pci.txt"
for label in baseline irq_cpu3_priority80 rx_checksum_off tx_checksum_off napi_timer_100us restored; do
  restore_parameters
  case "$label" in
    irq_cpu3_priority80)
      printf '3\n' > "/proc/irq/$irq/smp_affinity_list"
      chrt -f -p 80 "$irq_tid" ;;
    rx_checksum_off) "$eth" -K "$iface" rx off ;;
    tx_checksum_off) "$eth" -K "$iface" tx off ;;
    napi_timer_100us)
      printf '100000\n' > "/sys/class/net/$iface/gro_flush_timeout"
      printf '2\n' > "/sys/class/net/$iface/napi_defer_hard_irqs" ;;
  esac
  echo "START $label (ICMP and state reception only)"
  cat /proc/interrupts > "$out/${label}_interrupts_before.txt"
  cat /proc/net/snmp > "$out/${label}_snmp_before.txt"
  cat "/proc/irq/$irq/smp_affinity_list" > "$out/${label}_irq_affinity.txt"
  "$eth" -k "$iface" > "$out/${label}_features.txt"
  printf '\n' > "$instance/trace"
  taskset -c 0 /usr/sbin/tcpdump -Z "$operator_user" -i "$iface" -nn -s 0 -B 4096 \
    -w "$out/${label}.pcap" 'host 172.16.0.2 and (udp or icmp)' > "$out/${label}_capture.log" 2>&1 &
  capture_pid=$!
  sleep .2
  kill -0 "$capture_pid"
  printf '1\n' > "$instance/tracing_on"
  python3 -c 'import time; print(time.monotonic_ns(), time.time_ns())' > "$out/${label}_icmp_start_clock.txt"
  runuser -u "$operator_user" -- timeout 10 "$probe_dir/icmp_probe" 172.16.0.2 2341 4000 "$out/${label}" > "$out/${label}_icmp.log" 2>&1
  python3 -c 'import time; print(time.monotonic_ns(), time.time_ns())' > "$out/${label}_read_start_clock.txt"
  runuser -u "$operator_user" -- timeout 10 "$probe_dir/latency_probe" --read 3 4 "$out/${label}" 172.16.0.2 > "$out/${label}_read.log" 2>&1
  printf '0\n' > "$instance/tracing_on"
  stop_capture
  gzip -c "$instance/trace" > "$out/${label}_trace.txt.gz"
  cat /proc/interrupts > "$out/${label}_interrupts_after.txt"
  cat /proc/net/snmp > "$out/${label}_snmp_after.txt"
  cat "$out/${label}_icmp.log" "$out/${label}_read.log"
done
echo 'All comparison runs completed; restoring original parameters.'

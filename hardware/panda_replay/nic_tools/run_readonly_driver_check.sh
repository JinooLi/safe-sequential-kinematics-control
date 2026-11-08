#!/usr/bin/env bash
# One authenticated, fixed diagnostic job. NEVER calls control or recovery.
set -euo pipefail
[[ $EUID == 0 ]] || { echo "Run this fixed diagnostic job with sudo." >&2; exit 2; }
[[ $# == 0 ]] || { echo "This diagnostic accepts no arguments." >&2; exit 2; }
root=/home/cdsl/jinoo_ws/panda-hardware-replay
operator_user=cdsl
replayer="$root/build/panda_replay"
plan="$root/panda_cpp_replay_after_eee/replay.plan"
robot_ip=172.16.0.2
bash "$root/nic_tools/r8169_rt_fix/load_r8169_temporary.sh" --apply

run_dir=$(mktemp -d "$root/capture_readonly_driver_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
chown "$operator_user:$(id -gn "$operator_user")" "$run_dir"
cd "$run_dir"
echo "Reading robot states only; no control commands will be sent."
timeout --signal=INT 15 /usr/sbin/tcpdump -Z "$operator_user" -i enp4s0 -nn -s 0 -U \
  -w "$run_dir/control.pcap" "host $robot_ip and (udp or icmp)" \
  >"$run_dir/capture.log" 2>&1 &
capture_pid=$!
sleep .5
if ! ps -p "$capture_pid" -o pid= >/dev/null; then
  cat "$run_dir/capture.log" >&2
  exit 1
fi
set +e
runuser -u "$operator_user" -- "$replayer" --diagnose "$plan" "$robot_ip" 2>&1 | tee "$run_dir/read.log"
read_status=${PIPESTATUS[0]}
wait "$capture_pid"
capture_status=$?
set -e
echo "Read status: $read_status; capture status: $capture_status (124 = timed capture completed)"
echo "Capture directory: $run_dir"
cat "$run_dir/capture.log"
if [[ $capture_status != 0 && $capture_status != 124 ]]; then exit 1; fi
exit "$read_status"

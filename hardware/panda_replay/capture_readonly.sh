#!/usr/bin/env bash
# READ ONLY: Robot::read through --diagnose. Never starts control or recovery.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash capture_readonly.sh PLAN ROBOT_IP" >&2
  exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
replay_root=$(dirname -- "$script_dir")
replayer="$replay_root/build/panda_replay"
plan=$(realpath -- "$1")
robot_ip=$2
operator_user=$(id -un)
"$replayer" --preflight "$plan" "$robot_ip"
sudo -v
run_dir=$(mktemp -d "$replay_root/capture_readonly_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
echo "Capturing robot state reception only. No control commands will be sent."
sudo -n timeout --signal=INT 5 tcpdump -Z "$operator_user" -i enp4s0 -nn -s 0 -U \
  -w "$run_dir/control.pcap" "host $robot_ip and (udp or icmp)" \
  >"$run_dir/capture.log" 2>&1 &
capture_pid=$!
sleep .5
if ! ps -p "$capture_pid" -o pid= >/dev/null; then
  cat "$run_dir/capture.log" >&2
  exit 1
fi
set +e
"$replayer" --diagnose "$plan" "$robot_ip" 2>&1 | tee "$run_dir/read.log"
read_status=${PIPESTATUS[0]}
wait "$capture_pid"
capture_status=$?
set -e
echo "Capture directory: $run_dir"
echo "Read exit status: $read_status; capture exit status: $capture_status (124 = timed capture completed)"
cat "$run_dir/capture.log"
exit "$read_status"

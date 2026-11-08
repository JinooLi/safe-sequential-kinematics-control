#!/usr/bin/env bash
# Operator-run diagnostic. ACTIVATES joint position control through --hold-test.
# sudo is used only for a five-second packet capture; robot control runs as the user.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash capture_hold.sh PLAN ROBOT_IP" >&2
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
run_dir=$(mktemp -d "$replay_root/capture_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
echo "Capturing robot UDP timing. The following test ACTIVATES a position hold."
sudo -n timeout --signal=INT 5 tcpdump -Z "$operator_user" -i enp4s0 -nn -s 0 -U \
  -w "$run_dir/control.pcap" "host $robot_ip and (udp or icmp)" \
  >"$run_dir/capture.log" 2>&1 &
capture_pid=$!
sleep .5
# sudo changes the child's UID; kill -0 can report EPERM even while it is alive.
if ! ps -p "$capture_pid" -o pid= >/dev/null; then
  cat "$run_dir/capture.log" >&2
  exit 1
fi
set +e
"$replayer" --hold-test "$plan" "$robot_ip" 2>&1 | tee "$run_dir/hold.log"
hold_status=${PIPESTATUS[0]}
wait "$capture_pid"
capture_status=$?
set -e
echo "Capture directory: $run_dir"
echo "Hold exit status: $hold_status; capture exit status: $capture_status (124 = timed capture completed)"
cat "$run_dir/capture.log"
exit "$hold_status"

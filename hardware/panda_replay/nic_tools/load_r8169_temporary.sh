#!/usr/bin/env bash
# Operator-run, runtime-only driver comparison for this lab-server.
# Does not install a module or change boot configuration; never starts robot control.
set -euo pipefail
if [[ $EUID != 0 ]]; then
  echo "Run with sudo in the lab-server terminal." >&2
  exit 2
fi
mode=${1:---apply}
if [[ $# -gt 1 || ( $mode != --apply && $mode != --restore ) ]]; then
  echo "Usage: sudo bash load_r8169_temporary.sh [--apply|--restore]" >&2
  exit 2
fi
root=/home/cdsl/jinoo_ws/panda-hardware-replay
module="$root/nic_tools/r8169_rt_fix/source_no_thread/r8169.ko"
stock=/lib/modules/5.9.1-rt20/kernel/drivers/net/ethernet/realtek/r8169.ko
ethtool="$root/nic_tools/unpacked/sbin/ethtool"
iface=enp4s0
expected_sha=e293a2c7e3e35d0d92853908460f4a1c052056db215efcdd035af7837f56ddd6
[[ $(uname -r) == 5.9.1-rt20 ]] || { echo "Unexpected kernel; refusing." >&2; exit 1; }
if pgrep -x panda_replay >/dev/null; then
  echo "A panda_replay process is running; stop it before changing the NIC driver." >&2
  exit 1
fi
if [[ $mode == --apply ]]; then
  [[ $(sha256sum "$module" | cut -d ' ' -f1) == "$expected_sha" ]] || {
    echo "Patched module checksum mismatch." >&2; exit 1;
  }
  [[ $(/sbin/modinfo -F vermagic "$module") == $(/sbin/modinfo -F vermagic "$stock") ]] || {
    echo "Module kernel metadata mismatch." >&2; exit 1;
  }
  # Read-only check: Idle, current error-free, and the known start pose.
  "$root/build/panda_replay" --preflight "$root/panda_cpp_replay_after_eee/replay.plan" 172.16.0.2
fi
for device in /sys/class/net/*; do
  [[ -L $device/device/driver ]] || continue
  if [[ $(basename "$(readlink -f "$device/device/driver")") == r8169 && $(basename "$device") != "$iface" ]]; then
    echo "Another NIC uses r8169; refusing to unload a shared driver." >&2
    exit 1
  fi
done

configure_link() {
  local ready=0
  for attempt in {1..50}; do
    if [[ -d /sys/class/net/$iface ]]; then ready=1; break; fi
    sleep .1
  done
  (( ready == 1 )) || return 1
  ip link set dev "$iface" up || return 1
  ip address replace 172.16.0.1/24 dev "$iface" || return 1
  "$ethtool" --set-eee "$iface" eee off || return 1
  "$ethtool" -K "$iface" gro off || return 1
  for attempt in {1..100}; do
    if [[ $(cat "/sys/class/net/$iface/carrier") == 1 ]]; then return 0; fi
    sleep .1
  done
  echo "NIC link did not recover within 10 seconds." >&2
  return 1
}

echo "Temporarily reloading the robot NIC driver. No robot control will be started."
# Keep the installed module untouched. A failed load/configuration restores it.
if [[ -d /sys/module/r8169 ]]; then /sbin/rmmod r8169; fi
complete=0
restore_on_failure() {
  local status=$?
  if (( complete == 0 )); then
    trap - EXIT
    set +e
    echo "Restoring the installed driver after an incomplete switch." >&2
    if [[ -d /sys/module/r8169 ]]; then /sbin/rmmod r8169; fi
    /sbin/modprobe r8169
    configure_link
  fi
  exit "$status"
}
trap restore_on_failure EXIT
if [[ $mode == --apply ]]; then
  /sbin/insmod "$module"
  expected_version=$(/sbin/modinfo -F srcversion "$module")
else
  /sbin/modprobe r8169
  expected_version=$(/sbin/modinfo -F srcversion "$stock")
fi
configure_link
[[ $(cat /sys/module/r8169/srcversion) == "$expected_version" ]]
complete=1
echo "Loaded srcversion: $expected_version"
ip -brief address show dev "$iface"
echo "Runtime driver switch completed. System module files and boot settings are unchanged."

"""Exercise failure handling in isolation; never access NICs or the robot."""
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "nic_tools/check_napi_readonly.sh"


class NicScriptFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = SCRIPT.read_text()
        for name in ("sys/class/net/testnic/device/power", "sys/module/r8169", "proc/irq/127", "out"):
            (self.root / name).mkdir(parents=True)
        (self.root / "sys/module/r8169/srcversion").write_text("4910F4E779F70AFA2802017\n")
        (self.root / "out/before_routes.json").write_text("[]")
        (self.root / "out/before_features.txt").write_text("generic-receive-offload: off\n")

    def function(self, name):
        match = re.search(r"^" + name + r"\(\) \{\n.*?^\}", self.source, re.M | re.S)
        self.assertIsNotNone(match)
        text = match[0].replace("/sys/", str(self.root / "sys") + "/")
        text = text.replace("/proc/", str(self.root / "proc") + "/")
        return text.replace("/sbin/rmmod", "mock_rmmod").replace("/sbin/insmod", "mock_insmod")

    def run_shell(self, names, body):
        prefix = f"sandbox={shlex.quote(str(self.root))}\nout=\"$sandbox/out\"\n"
        prefix += "iface=testnic\noriginal_power=on\noperator_user=unused\nprobe=unused\n"
        result = subprocess.run(
            ["bash", "-c", prefix + "\n".join(self.function(n) for n in names) + "\n" + body],
            text=True, capture_output=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_failed_state_read_does_not_parse_empty_json(self):
        result = self.run_shell(["state_check"], r'''
runuser() { echo "connection unavailable" >&2; return 1; }
sleep() { :; }
if state_check "$out/state.json"; then exit 1; fi
''')
        self.assertIn("connection unavailable", result.stderr)
        self.assertNotIn("JSONDecodeError", result.stderr)

    def test_feature_failure_still_waits_for_carrier_and_restores_power(self):
        (self.root / "sys/class/net/testnic/carrier").write_text("0\n")
        self.run_shell(["configure_link"], r'''
ip() { :; }
fake_eth() {
  if [[ $1 == -K ]]; then return 1; fi
  if [[ $1 == -k ]]; then cat "$out/before_features.txt"; fi
  return 0
}
eth=fake_eth
sleep() { printf '1\n' > "$sandbox/sys/class/net/testnic/carrier"; }
if configure_link; then exit 1; fi
[[ $(cat "$sandbox/sys/class/net/testnic/carrier") == 1 ]]
[[ $(cat "$sandbox/sys/class/net/testnic/device/power/control") == on ]]
[[ -f $out/current_features_diff.txt ]]
''')

    def test_stock_reload_restores_parameters_even_when_link_setup_fails(self):
        self.run_shell(["load_stock"], r'''
mock_rmmod() { :; }
mock_insmod() { :; }
configure_link() { return 1; }
restore_parameters() { touch "$out/restored_parameters"; }
stock=unused
driver_touched=1
if load_stock; then exit 1; fi
[[ -f $out/restored_parameters && $driver_touched == 0 ]]
''')

    def test_priority_failure_does_not_skip_napi_restoration(self):
        self.run_shell(["restore_parameters"], r'''
find_irq() { irq=127; tid=unused; }
chrt() { return 1; }
original_affinity=21
original_priority=50
original_gro_timeout=0
original_defer=0
if restore_parameters; then exit 1; fi
[[ $(cat "$sandbox/proc/irq/127/smp_affinity_list") == 21 ]]
[[ $(cat "$sandbox/sys/class/net/testnic/gro_flush_timeout") == 0 ]]
[[ $(cat "$sandbox/sys/class/net/testnic/napi_defer_hard_irqs") == 0 ]]
''')

    def test_vendor_module_is_unloaded_before_loading_stock(self):
        (self.root / "sys/module/r8125").mkdir()
        self.run_shell(["load_stock"], r'''
vendor=1
mock_rmmod() { printf 'unload %s\n' "$1" >> "$out/module_actions"; }
mock_insmod() { printf 'load %s\n' "$1" >> "$out/module_actions"; }
configure_link() { :; }
restore_parameters() { :; }
stock=original_module
driver_touched=1
load_stock || exit 1
[[ $(head -1 "$out/module_actions") == 'unload r8125' ]]
[[ $(tail -1 "$out/module_actions") == 'load original_module' ]]
[[ $driver_touched == 0 ]]
''')

    def test_failed_vendor_unload_does_not_load_a_competing_driver(self):
        (self.root / "sys/module/r8125").mkdir()
        self.run_shell(["load_stock"], r'''
vendor=1
mock_rmmod() { return 1; }
mock_insmod() { touch "$out/unexpected_load"; }
configure_link() { :; }
restore_parameters() { :; }
stock=unused
driver_touched=1
if load_stock; then exit 1; fi
[[ ! -e $out/unexpected_load && $driver_touched == 1 ]]
''')


if __name__ == "__main__":
    unittest.main()

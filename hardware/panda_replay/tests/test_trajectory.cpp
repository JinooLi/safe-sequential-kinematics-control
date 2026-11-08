#include "trajectory.hpp"
#include "timing.hpp"
#include "control_health.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace panda_replay;
namespace {
void expect(bool value, const char* what) { if (!value) throw std::runtime_error(what); }
template<class F> void rejects(F operation) {
  bool rejected = false;
  try { operation(); } catch (const std::runtime_error&) { rejected = true; }
  expect(rejected, "Invalid input was accepted");
}
}
int main() {
  try {
    Segment s;
    s.duration = 10;
    for (unsigned j = 0; j < 7; ++j) s.c[j] = {{0, 0, 0, 10, -15, 6}};
    expect(std::abs(s.value(.5)[0] - .5) < 1e-12, "Quintic midpoint");
    expect(std::abs(s.value(.5, 1)[0] - .1875) < 1e-12, "Physical-time derivative");
    expect(s.bound(1)[0] >= .1875, "Bernstein bounds must enclose interior peak");
    expect(std::abs(s.value(1, 1)[0]) < 1e-12 && std::abs(s.value(1, 2)[0]) < 1e-12, "Rest endpoint");
    for (unsigned d = 0; d <= 3; ++d)
      for (int sample = 0; sample <= 1000; ++sample)
        expect(std::abs(s.value(sample / 1000., d)[0]) <= s.bound(d)[0] + 1e-12,
               "Continuous bound underestimated derivative");
    // A peak that endpoint-only validation would miss.
    s.c[0] = {{0, 4, -4, 0, 0, 0}};
    expect(s.value(0)[0] == 0 && s.value(1)[0] == 0 && s.bound(0)[0] >= 1,
           "Interior overshoot not bounded");
    Plan p;
    s.c[0] = {{0, 0, 0, 10, -15, 6}};
    p.segments = {s}; p.ends = {10}; p.start_budget = .0001; p.tracking = .005;
    Joints zero{}, near{}; near[1] = .00005;
    expect(std::abs(start_offset(p, near, near, zero, zero, zero)[1] - .00005) < 1e-12,
           "Small stationary offset");
    near[1] = .0002;
    rejects([&] { start_offset(p, near, near, zero, zero, zero); });
    near = zero; near[0] = .02;
    rejects([&] { start_offset(p, zero, zero, near, zero, zero); });
    rejects([&] { start_offset(p, zero, near, zero, zero, zero); });
    expect(p.value(-1) == p.value(0) && p.value(11) == p.value(10), "Time endpoint clamp");
    TimingTrace trace;
    for (std::size_t i = 0; i < 1100; ++i)
      trace.record({i, double(i), i == 1099 ? 3. : 1., 0., 1., 20.});
    expect(trace.size() == 1100 && trace.over_2ms() == 1, "Timing summary lost wrapped samples");
    expect(trace.at(76).index == 76 && trace.at(1099).index == 1099,
           "Timing ring must retain the last 1024 samples");
    try {
      TimingTrace::Scope timing(trace, 1200, 4, 1., 1.);
      throw std::runtime_error("Simulated callback fault");
    } catch (const std::runtime_error&) {}
    expect(trace.size() == 1101 && trace.at(1100).period_ms == 4., "Faulting callback missing from trace");
    using Status = CommunicationGate::Status;
    CommunicationGate healthy;
    expect(healthy.update(0, 0.) == Status::waiting, "Initial empty rate window");
    for (unsigned i = 1; i < 500; ++i)
      expect(healthy.update(1, i < 100 ? 0. : 1.) == Status::waiting, "Must hold before 500 ms");
    expect(healthy.update(1, 1.) == Status::ready, "Healthy connection can start at 500 ms");
    expect(healthy.update(1, .99) == Status::ready, "One lost command in rolling 100-command window");
    expect(healthy.update(1, .98) == Status::poor_reception, "Reception guard must remain active during playback");
    CommunicationGate poor;
    poor.update(0, 0.);
    for (unsigned i = 1; i < 100; ++i) poor.update(1, .57);
    expect(poor.update(1, .57) == Status::poor_reception, "Observed failure must stop while holding start pose");
    CommunicationGate invalid;
    expect(invalid.update(0, 1.) == Status::waiting, "First zero period");
    expect(invalid.update(0, 1.) == Status::invalid, "Repeated zero period");
    expect(invalid.update(3, 1.) == Status::invalid, "Communication guard preserves 2 ms limit");
    // Reproduce old experiment timing gaps and the observed low reception. In
    // warn mode neither prevents the 500 ms startup hold from completing.
    CommunicationGate warn(CommunicationGate::Policy::warn);
    expect(warn.update(0, 0.) == Status::waiting, "Warn policy initial sample");
    for (unsigned i = 0; i < 99; ++i)
      expect(warn.update(5, .56) == Status::waiting, "Warn policy must not start playback early");
    expect(warn.elapsed_ms() == 495, "Warn policy uses robot elapsed time");
    expect(warn.update(5, .56) == Status::ready, "Warn policy permits observed low reception after hold");
    expect(warn.gaps_over_2ms() == 100 && warn.poor_reception_samples() == 81 &&
           std::abs(warn.min_success() - .56) < 1e-12, "Communication warning counts must respect warmup");
    expect(warn.update(4, .5) == Status::ready, "Warn policy stays ready during playback");
    const auto valid_samples = warn.samples();
    expect(warn.update(0, 1.) == Status::invalid, "Warn policy still rejects repeated zero periods");
    for (double rate : {-0.01, 1.01, std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity()})
      expect(warn.update(1, rate) == Status::invalid, "Warn policy must reject invalid rate feedback");
    expect(warn.samples() == valid_samples, "Rejected feedback must not advance startup clock");
    // The actual previous replay failure should warn only when explicitly selected.
    using DesiredStatus = DesiredStateGate::Status;
    Joints desired{};
    DesiredStateGate strict_desired;
    desired[6] = .001;
    expect(strict_desired.update(zero, desired) == DesiredStatus::ok, "Desired threshold boundary");
    desired[6] = -.0010316597725580356;
    expect(strict_desired.update(zero, desired) == DesiredStatus::deviation,
           "Default policy must retain the observed desired-state stop");
    DesiredStateGate warn_desired(DesiredStateGate::Policy::warn);
    for (unsigned i = 0; i < 1100; ++i)
      expect(warn_desired.update(zero, desired) == DesiredStatus::ok,
             "Warn policy must permit the previous replay deviation");
    expect(warn_desired.exceeded_samples() == 1100 && warn_desired.max_joint() == 7 &&
           std::abs(warn_desired.max_deviation() - .0010316597725580356) < 1e-12,
           "Desired warning totals must survive trace ring wrap");
    desired[6] = 0;
    expect(warn_desired.update(zero, desired) == DesiredStatus::ok &&
           warn_desired.exceeded_samples() == 1100, "Warnings must count only deviations");
    for (double value : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
      desired[0] = value;
      expect(warn_desired.update(zero, desired) == DesiredStatus::invalid,
             "Warning policy must reject non-finite desired state");
      expect(warn_desired.update(desired, zero) == DesiredStatus::invalid,
             "Warning policy must reject non-finite reference");
    }
    std::cout << "Trajectory core tests passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

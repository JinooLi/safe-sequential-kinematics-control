#include "trajectory.hpp"
#include "timing.hpp"
#include "control_health.hpp"
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef HAVE_FRANKA
#include <franka/control_types.h>
#include <franka/control_tools.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <sys/mman.h>
#endif

namespace {
#ifdef HAVE_FRANKA
volatile std::sig_atomic_t stop_requested = 0;
void request_stop(int) { stop_requested = 1; }
#endif
void usage() {
  std::cerr << "Usage:\n"
            << "  panda_replay --check PLAN\n"
            << "  panda_replay --sample PLAN TIME  (offline JSON position and floor clearance)\n"
            << "  panda_replay --snapshot ROBOT_IP  (read-only JSON on stdout)\n"
            << "  panda_replay --recover-communication ROBOT_IP  (explicit communication-reflex reset only)\n"
            << "  panda_replay --preflight PLAN ROBOT_IP  (read-only live start check)\n"
            << "  panda_replay --diagnose PLAN ROBOT_IP  (10-second read-only timing test)\n"
            << "  panda_replay --hold-test PLAN ROBOT_IP [OPTIONS]  (10-second ACTIVE position hold)\n"
            << "  panda_replay --execute PLAN ROBOT_IP [OPTIONS]  (MOVES THE ROBOT)\n"
            << "  --communication-policy=warn: log >2 ms gaps and <99% reception without cancelling.\n"
            << "  --desired-state-policy=warn: log desired-position deviation >0.001 rad without cancelling.\n"
            << "  Both policies default to strict. Measured tracking and floor checks remain active.\n";
}
void desired_state_summary(const panda_replay::DesiredStateGate& gate) {
  if (gate.policy() != panda_replay::DesiredStateGate::Policy::warn) return;
  std::cerr << "Desired-state warnings: samples_over_0.001rad=" << gate.exceeded_samples()
            << ", max_deviation_rad=" << gate.max_deviation()
            << ", joint=" << gate.max_joint()
            << ". Desired-state deviation did not request cancellation under the warn policy.\n";
}
void communication_summary(const panda_replay::CommunicationGate& gate) {
  if (gate.policy() != panda_replay::CommunicationGate::Policy::warn || !gate.samples()) return;
  std::cerr << "Communication warnings: gaps_over_2ms=" << gate.gaps_over_2ms()
            << ", low_reception_samples_after_100ms=" << gate.poor_reception_samples()
            << ", min_success_rate_after_100ms=" << gate.min_success()
            << ". These conditions did not request cancellation under the warn policy.\n";
}
void array(std::ostream& out, const panda_replay::Joints& q) {
  out << '[';
  for (size_t j = 0; j < q.size(); ++j) out << (j ? "," : "") << q[j];
  out << ']';
}
#ifdef HAVE_FRANKA
void idle(const franka::RobotState& state) {
  if (state.robot_mode != franka::RobotMode::kIdle || bool(state.current_errors))
    throw std::runtime_error("Robot must be Idle and error-free; check Desk/FCI");
  // This model covers the stock Panda hand with no grasped object or added tool.
  if (std::abs(state.m_load) > .001 || std::abs(state.m_ee - .73) > .01)
    throw std::runtime_error("Configured payload differs from the unloaded stock Panda hand");
}
void snapshot(const franka::RobotState& state) {
  idle(state);
  std::cout << std::setprecision(17) << "{\"robot_mode\":\"Idle\",\"current_errors\":[],\"q\":";
  array(std::cout, state.q);
  std::cout << ",\"q_d\":"; array(std::cout, state.q_d);
  std::cout << ",\"dq\":"; array(std::cout, state.dq);
  std::cout << ",\"dq_d\":"; array(std::cout, state.dq_d);
  std::cout << ",\"ddq_d\":"; array(std::cout, state.ddq_d);
  std::cout << ",\"robot_time_ms\":" << state.time.toMSec() << "}\n";
}
#endif
}  // namespace

int main(int argc, char** argv) {
  panda_replay::TimingTrace trace;
  panda_replay::CommunicationGate communication;
  panda_replay::DesiredStateGate desired_state;
  const char* trace_mode = "control";
  try {
    if (argc < 3) { usage(); return 2; }
    const std::string mode = argv[1];
    if (mode == "--recover-communication" && argc == 3) {
#ifdef HAVE_FRANKA
      franka::Robot robot(argv[2]);
      const auto state = robot.readOnce();
      if (state.robot_mode != franka::RobotMode::kIdle) {
        if (state.robot_mode != franka::RobotMode::kReflex || bool(state.current_errors) ||
            !state.last_motion_errors.communication_constraints_violation)
          throw std::runtime_error("Recovery mode only accepts a communication Reflex with no active errors; inspect Desk");
        // Reject a reflex with any additional error, especially contact/collision.
        const std::string errors = static_cast<std::string>(state.last_motion_errors);
        if (errors.find(',') != std::string::npos)
          throw std::runtime_error("Additional motion errors present; inspect Desk before recovery");
        robot.automaticErrorRecovery();
      }
      snapshot(robot.readOnce());
      return 0;  // Explicit operator request only; never enters robot.control().
#else
      throw std::runtime_error("Built without libfranka; only offline modes are available");
#endif
    }
    if (mode == "--snapshot" && argc == 3) {
#ifdef HAVE_FRANKA
      franka::Robot robot(argv[2]);
      snapshot(robot.readOnce());
      return 0;
#else
      throw std::runtime_error("Built without libfranka; only offline modes are available");
#endif
    }
    const bool active_mode = mode == "--execute" || mode == "--hold-test";
    bool warn_communication = false, warn_desired_state = false;
    bool valid_active_options = active_mode && argc >= 4 && argc <= 6;
    if (valid_active_options) for (int i = 4; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--communication-policy=warn" && !warn_communication) warn_communication = true;
      else if (option == "--desired-state-policy=warn" && !warn_desired_state) warn_desired_state = true;
      else valid_active_options = false;
    }
    if (!((mode == "--check" && argc == 3) || (mode == "--sample" && argc == 4) ||
          ((mode == "--preflight" || mode == "--diagnose") && argc == 4) ||
          valid_active_options)) {
      usage(); return 2;
    }
    if (warn_communication)
      communication = panda_replay::CommunicationGate(panda_replay::CommunicationGate::Policy::warn);
    if (warn_desired_state)
      desired_state = panda_replay::DesiredStateGate(panda_replay::DesiredStateGate::Policy::warn);
    auto plan = panda_replay::Plan::load(argv[2]);
    if (mode == "--sample") {
      size_t consumed = 0;
      const double t = std::stod(argv[3], &consumed);
      if (consumed != std::string(argv[3]).size() || !std::isfinite(t) || t < 0 || t > plan.duration())
        throw std::runtime_error("Sample time must be inside the plan");
      auto q = plan.value(t);
      std::cout << std::setprecision(17) << "{\"q\":"; array(std::cout, q);
      std::cout << ",\"floor_clearance\":" << plan.floor_clearance(q) << "}\n";
      return 0;
    }
    plan.validate();  // File parsing, geometry checks and allocations happen before FCI control.
    if (mode == "--check") return 0;
#ifdef HAVE_FRANKA
    franka::Robot robot(argv[3]);  // Default RealtimeConfig::kEnforce.
    const auto before = robot.readOnce();
    idle(before);
    if (mode == "--diagnose") {
      trace_mode = "read-only; command path not tested";
      std::string error;
      if (!franka::hasRealtimeKernel() || !franka::setCurrentThreadToHighestSchedulerPriority(&error))
        throw std::runtime_error("Read-only timing test requires realtime scheduling: " + error);
      if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) throw std::runtime_error("mlockall failed; check the realtime memlock limit");
      std::cout << "Reading robot states for 10 seconds; no control or motion commands.\n" << std::flush;
      double first_time = -1, previous_time = 0;
      volatile double computation_sink = 0;
      robot.read([&](const franka::RobotState& state) -> bool {
        const double now = state.time.toSec();
        const double period_ms = first_time < 0 ? 0. : (now - previous_time) * 1000.;
        if (first_time < 0) first_time = now;
        previous_time = now;
        // Millisecond integer timestamps avoid subtraction roundoff at 1 and 2 ms.
        const double rounded_ms = std::round(period_ms);
        panda_replay::TimingTrace::Scope timing(trace, state.time.toMSec(), rounded_ms,
                                                now - first_time, state.control_command_success_rate);
        const auto reference = plan.value(std::min(now - first_time, plan.duration()));
        computation_sink = plan.floor_clearance(state.q) + reference[0];
        return now - first_time < 10.;
      });
      (void)computation_sink;
      munlockall();
      trace.dump(trace_mode);
      std::cout << "Read-only timing test finished. This does not verify command delivery during motion.\n";
      return trace.over_2ms() ? 1 : 0;
    }
    auto offset = panda_replay::start_offset(plan, before.q_d, before.q, before.dq, before.dq_d, before.ddq_d);
    if (mode == "--preflight") {
      std::cout << "Live start state matches the plan; no motion was commanded.\n";
      return 0;
    }
    std::ifstream rt("/sys/kernel/realtime");
    int realtime = 0;
    if (!(rt >> realtime) || realtime != 1) throw std::runtime_error("PREEMPT_RT kernel is required");
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) throw std::runtime_error("mlockall failed; check the realtime memlock limit");
    std::signal(SIGINT, request_stop);
    std::signal(SIGTERM, request_stop);
    const bool hold_only = mode == "--hold-test";
    trace_mode = hold_only ? "active position hold" : "control";
    std::cout << (hold_only ? "Active position hold for 10 seconds; sending FCI commands.\n" :
                             "Holding the start pose for 0.5 s before replay.\n")
              << (warn_communication ? "Communication policy: warn; >2 ms gaps and <99% reception are logged without cancellation.\n" :
                                       "Communication policy: strict; requires <=2 ms gaps and >=99% reception after 100 ms.\n")
              << (warn_desired_state ? "Desired-state policy: warn; >0.001 rad desired deviation is logged; measured tracking and floor checks remain active.\n" :
                                       "Desired-state policy: strict; desired deviation >0.001 rad cancels control.\n")
              << "Ctrl-C requests controller cancellation.\n" << std::flush;
    double time = 0;
    bool first = true;
    bool playback_started = false;
    auto previous = before.q_d;
    // No robot safety thresholds, gains, payload, or recovery settings are changed.
    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
      panda_replay::TimingTrace::Scope timing(trace, state.time.toMSec(), period.toMSec(),
                                              time, state.control_command_success_rate);
      timing.feedback(previous, state.q, state.q_d);
      if (stop_requested) throw std::runtime_error("Operator requested cancellation");
      if (bool(state.current_errors)) throw std::runtime_error("Robot reported an error");
      const double elapsed = period.toSec();
      if (first) {
        // Recheck inside the callback to close the readOnce -> control start race.
        offset = panda_replay::start_offset(plan, state.q_d, state.q, state.dq, state.dq_d, state.ddq_d);
        first = false;
      } else {
        if (!(elapsed > 0)) throw std::runtime_error("Invalid FCI callback period");
        if (!warn_communication && elapsed > .002)
          throw std::runtime_error("FCI callback gap exceeds 2 ms");
      }
      const auto health = communication.update(period.toMSec(), state.control_command_success_rate);
      if (health == panda_replay::CommunicationGate::Status::invalid)
        throw std::runtime_error("Invalid FCI communication feedback");
      if (health == panda_replay::CommunicationGate::Status::poor_reception)
        throw std::runtime_error("FCI command reception below 99%; inspect network before replay");
      if (!hold_only && health == panda_replay::CommunicationGate::Status::ready) {
        if (playback_started) time += elapsed;
        else playback_started = true;
      }
      for (unsigned j = 0; j < 7; ++j) {
        if (!std::isfinite(state.q[j]) || std::abs(state.q[j] - previous[j]) > plan.tracking)
          throw std::runtime_error("Measured joint tracking error exceeds plan allowance");
      }
      const auto desired_health = desired_state.update(previous, state.q_d);
      if (desired_health == panda_replay::DesiredStateGate::Status::invalid)
        throw std::runtime_error("Invalid FCI desired-state feedback");
      if (desired_health == panda_replay::DesiredStateGate::Status::deviation)
        throw std::runtime_error("FCI desired-state deviation from planned command exceeds 0.001 rad");
      if (plan.floor_clearance(state.q) < plan.floor_margin)
        throw std::runtime_error("Measured pose violates model floor margin");
      auto q = plan.value(std::min(time, plan.duration()));
      for (unsigned j = 0; j < 7; ++j) q[j] += offset[j];
      previous = q;
      franka::JointPositions command(q);
      if ((hold_only && communication.elapsed_ms() >= 10000) || (!hold_only && time >= plan.duration()))
        return franka::MotionFinished(command);
      return command;
    }, franka::ControllerMode::kJointImpedance, true, franka::kMaxCutoffFrequency);
    munlockall();
    trace.dump(trace_mode);
    communication_summary(communication);
    desired_state_summary(desired_state);
    std::cout << (hold_only ? "Active hold test passed under the selected communication policy; no trajectory playback was requested.\n" :
                             "Replay finished at the validated zero-velocity endpoint.\n");
    return 0;
#else
    throw std::runtime_error("Built without libfranka; rebuild on the robot host with PANDA_WITH_FRANKA=ON");
#endif
  } catch (const std::exception& error) {
    std::cerr << "panda_replay: " << error.what() << '\n';
    trace.dump(trace_mode);
    communication_summary(communication);
    desired_state_summary(desired_state);
    return 1;  // Never automatically recover or restart a cancelled hardware motion.
  }
}

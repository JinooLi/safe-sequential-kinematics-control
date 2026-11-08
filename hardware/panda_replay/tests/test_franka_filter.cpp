#include "trajectory.hpp"
#include <franka/rate_limiting.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Pure numerical test: no Robot object, network connection, or motion command.
// Models perfect 1 kHz delivery with the previous limited command as q_d.
int main(int argc, char** argv) {
  try {
    panda_replay::Plan plan;
    if (argc == 2) {
      plan = panda_replay::Plan::load(argv[1]);
    } else {
      panda_replay::Segment segment;
      segment.duration = 4.;
      const panda_replay::Joints start{{0, -.7, 0, -2.3, 0, 1.5, .7}};
      for (unsigned j = 0; j < 7; ++j)
        segment.c[j] = {{start[j], 0, 0, .1, -.15, .06}};
      plan.segments = {segment};
      plan.ends = {segment.duration};
    }
    auto position = plan.value(0);
    panda_replay::Joints velocity{}, acceleration{};
    double max_change = 0;
    const double dt = .001;
    const auto count = static_cast<std::size_t>(std::ceil(plan.duration() / dt));
    for (std::size_t i = 1; i <= count; ++i) {
      const auto command = plan.value(std::min(i * dt, plan.duration()));
      const auto limited = franka::limitRate(franka::kMaxJointVelocity, franka::kMaxJointAcceleration,
                                             franka::kMaxJointJerk, command, position, velocity, acceleration);
      for (unsigned j = 0; j < 7; ++j) {
        max_change = std::max(max_change, std::abs(command[j] - limited[j]));
        const double next_velocity = (limited[j] - position[j]) / dt;
        acceleration[j] = (next_velocity - velocity[j]) / dt;
        velocity[j] = next_velocity;
      }
      position = limited;
    }
    std::cout << "Offline libfranka rate limiter: " << count << " perfect-delivery samples; max modification="
              << max_change << " rad\n";
    if (max_change > 1e-8) throw std::runtime_error("Smooth trajectory unexpectedly modified under perfect delivery");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#pragma once
#include <array>
#include <string>
#include <vector>

namespace panda_replay {
using Joints = std::array<double, 7>;
using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;
constexpr double kVelocity = .15;
constexpr double kAcceleration = 2.;
constexpr double kJerk = 100.;
constexpr Joints kLower{{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973}};
constexpr Joints kUpper{{2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973}};

struct Segment {
  double duration{};
  std::array<std::array<double, 6>, 7> c{};  // Ascending powers of s=t/duration.
  Joints value(double s, unsigned derivative = 0) const;
  Joints bound(unsigned derivative) const;
};
struct Body {
  int parent{}, joint{-1};
  Vec3 position{};
  Mat3 rotation{};
  std::vector<Vec3> points;
  Joints radii{};
};
struct Plan {
  std::vector<Segment> segments;
  std::vector<double> ends;
  std::vector<Body> bodies;  // World is index 0.
  double floor_z{}, floor_margin{}, self_margin{}, tracking{}, start_budget{}, geometry_dt{};
  double duration() const { return ends.back(); }
  Joints value(double t, unsigned derivative = 0) const;
  // No allocation: suitable for a callback. uncertainty is per-joint in radians.
  double floor_clearance(const Joints& q, const Joints& uncertainty = {}) const;
  void validate() const;
  static Plan load(const std::string& path);
};
// Returns a bounded constant alignment offset; never generates an unvalidated move.
Joints start_offset(const Plan& plan, const Joints& desired, const Joints& measured,
                    const Joints& velocity, const Joints& desired_velocity,
                    const Joints& desired_acceleration);
}  // namespace panda_replay

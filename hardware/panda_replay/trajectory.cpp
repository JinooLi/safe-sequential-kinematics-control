#include "trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace panda_replay {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
double number(std::istream& in) {
  double x;
  require(bool(in >> x) && std::isfinite(x), "Missing or non-finite number in plan");
  return x;
}
int integer(std::istream& in, int low, int high) {
  const double x = number(in);
  require(x >= low && x <= high && x == std::floor(x), "Invalid integer in plan");
  return static_cast<int>(x);
}
double choose(unsigned n, unsigned k) {
  double result = 1.;
  for (unsigned i = 1; i <= k; ++i) result *= double(n + 1 - i) / i;
  return result;
}
double multiplier(unsigned k, unsigned d) {
  double result = 1.;
  for (unsigned i = 0; i < d; ++i) result *= k - i;
  return result;
}
std::array<double, 6> bernstein(const Segment& segment, unsigned j, unsigned d) {
  std::array<double, 6> b{};
  for (unsigned i = 0; i <= 5 - d; ++i)
    for (unsigned k = 0; k <= i; ++k)
      b[i] += segment.c[j][k + d] * multiplier(k + d, d) * choose(i, k) /
              choose(5 - d, k) / std::pow(segment.duration, d);
  return b;
}
Mat3 identity() { return {{1, 0, 0, 0, 1, 0, 0, 0, 1}}; }
Mat3 multiply(const Mat3& a, const Mat3& b) {
  Mat3 c{};
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < 3; ++j)
      for (unsigned k = 0; k < 3; ++k) c[3*i+j] += a[3*i+k] * b[3*k+j];
  return c;
}
Vec3 multiply(const Mat3& a, const Vec3& b) {
  Vec3 c{};
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned k = 0; k < 3; ++k) c[i] += a[3*i+k] * b[k];
  return c;
}
double norm(const Vec3& v) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
Mat3 quaternion(std::istream& in) {
  double w = number(in), x = number(in), y = number(in), z = number(in);
  const double n = std::sqrt(w*w + x*x + y*y + z*z);
  require(std::abs(n - 1) < 1e-6, "Invalid body quaternion");
  w /= n; x /= n; y /= n; z /= n;
  return {{1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
           2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
           2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}};
}
}  // namespace

Joints Segment::value(double s, unsigned d) const {
  Joints result{};
  if (d > 5) return result;
  for (unsigned j = 0; j < 7; ++j) {
    for (int k = 5; k >= static_cast<int>(d); --k)
      result[j] = result[j] * s + c[j][k] * multiplier(k, d);
    result[j] /= std::pow(duration, d);
  }
  return result;
}
Joints Segment::bound(unsigned d) const {
  Joints result{};
  if (d > 5) return result;
  for (unsigned j = 0; j < 7; ++j)
    for (const double b : bernstein(*this, j, d)) result[j] = std::max(result[j], std::abs(b));
  return result;
}
Joints Plan::value(double t, unsigned d) const {
  const auto it = std::upper_bound(ends.begin(), ends.end(), t);
  const auto index = std::min(static_cast<size_t>(it - ends.begin()), ends.size() - 1);
  const double start = index == 0 ? 0 : ends[index - 1];
  return segments[index].value(std::max(0., std::min(1., (t - start) / segments[index].duration)), d);
}
double Plan::floor_clearance(const Joints& q, const Joints& uncertainty) const {
  std::array<Mat3, 12> rotations{};
  std::array<Vec3, 12> positions{};
  rotations[0] = identity();
  double clearance = std::numeric_limits<double>::infinity();
  for (size_t i = 1; i < bodies.size(); ++i) {
    const auto& body = bodies[i];
    positions[i] = multiply(rotations[body.parent], body.position);
    for (unsigned k = 0; k < 3; ++k) positions[i][k] += positions[body.parent][k];
    rotations[i] = multiply(rotations[body.parent], body.rotation);
    if (body.joint >= 0) {
      const double c = std::cos(q[body.joint]), s = std::sin(q[body.joint]);
      rotations[i] = multiply(rotations[i], Mat3{{c,-s,0,s,c,0,0,0,1}});
    }
    double padding = 0;
    bool moving = false;
    for (unsigned j = 0; j < 7; ++j) {
      padding += body.radii[j] * uncertainty[j];
      moving = moving || body.radii[j] > 0;
    }
    if (!moving) continue;  // Fixed mounting base rests on the mounting plane.
    for (const auto& point : body.points) {
      const double z = rotations[i][6]*point[0] + rotations[i][7]*point[1] +
                       rotations[i][8]*point[2] + positions[i][2];
      clearance = std::min(clearance, z - floor_z - padding);
    }
  }
  return clearance;
}

Plan Plan::load(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require(bool(file), "Cannot open plan: " + path);
  require(file.tellg() > 0 && file.tellg() < 32 * 1024 * 1024, "Plan size out of bounds");
  file.seekg(0);
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const size_t checksum = text.rfind("CHECKSUM64 ");
  require(checksum != std::string::npos && checksum > 0 && text[checksum-1] == '\n', "Missing checksum");
  std::uint64_t actual = UINT64_C(14695981039346656037), expected;
  for (size_t i = 0; i < checksum; ++i) actual = (actual ^ static_cast<unsigned char>(text[i])) * UINT64_C(1099511628211);
  std::istringstream trailer(text.substr(checksum + 11));
  require(bool(trailer >> std::hex >> expected) && expected == actual, "Plan checksum mismatch");
  trailer >> std::ws;
  require(trailer.eof(), "Unexpected data after checksum");
  std::istringstream in(text.substr(0, checksum));
  std::string magic;
  in >> magic;
  require(magic == "PANDA_REPLAY_V1", "Unsupported plan version");
  const int count = integer(in, 1, 100000), body_count = integer(in, 11, 11);
  Plan p;
  p.floor_z = number(in); p.floor_margin = number(in); p.self_margin = number(in);
  p.tracking = number(in); p.start_budget = number(in); p.geometry_dt = number(in);
  require(p.floor_margin >= .01 && p.self_margin >= .001 && p.tracking > 0 && p.tracking <= .005 &&
          p.start_budget >= 0 && p.start_budget <= .0001 && p.geometry_dt > 0 && p.geometry_dt <= .01,
          "Plan guard parameters are outside supported bounds");
  p.bodies.resize(body_count + 1);
  std::array<bool, 7> joints{};
  for (int i = 1; i <= body_count; ++i) {
    auto& b = p.bodies[i];
    b.parent = integer(in, 0, i - 1); b.joint = integer(in, -1, 6);
    const int points = integer(in, 4, 10000);
    for (auto& x : b.position) x = number(in);
    require(norm(b.position) < .5, "Unexpected Panda body translation");
    b.rotation = quaternion(in);
    if (b.joint >= 0) {
      require(!joints[b.joint], "Duplicate joint in geometry");
      joints[b.joint] = true;
    }
    b.points.resize(points);
    double radius = 0;
    for (auto& point : b.points) {
      for (auto& x : point) x = number(in);
      radius = std::max(radius, norm(point));
    }
    require(radius > .005 && radius < .6, "Unexpected Panda mesh extent");
    for (int ancestor = i; ancestor > 0; ancestor = p.bodies[ancestor].parent) {
      const auto& a = p.bodies[ancestor];
      if (a.joint >= 0) b.radii[a.joint] = radius;
      radius += norm(a.position);
    }
  }
  for (bool found : joints) require(found, "Missing joint in geometry");
  p.segments.resize(count);
  double total = 0;
  for (auto& s : p.segments) {
    s.duration = number(in);
    require(s.duration >= .001 && s.duration <= 300, "Segment duration out of bounds");
    for (auto& j : s.c) for (auto& x : j) x = number(in);
    total += s.duration;
    p.ends.push_back(total);
  }
  require(total <= 1800, "Plan exceeds 30 minutes");
  in >> std::ws;
  require(in.eof(), "Unexpected data before checksum");
  return p;
}

void Plan::validate() const {
  require(!segments.empty(), "Empty trajectory");
  Joints velocity{};
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& s = segments[i];
    for (unsigned j = 0; j < 7; ++j) {
      for (double b : bernstein(s, j, 0))
        require(b >= kLower[j] + .02 && b <= kUpper[j] - .02,
                "Spline position bound violates joint margin");
    }
    for (unsigned d = 1; d <= 3; ++d) {
      const auto bounds = s.bound(d);
      const double limit = d == 1 ? kVelocity : (d == 2 ? kAcceleration : kJerk);
      for (unsigned j = 0; j < 7; ++j) {
        require(std::isfinite(bounds[j]) && bounds[j] <= limit * (1 + 1e-9), "Spline derivative bound exceeds replay limits");
        if (d == 1) velocity[j] = std::max(velocity[j], bounds[j]);
      }
    }
    if (i > 0) for (unsigned d = 0; d <= 2; ++d) {
      auto previous = segments[i - 1].value(1., d), next = s.value(0., d);
      for (unsigned j = 0; j < 7; ++j)
        require(std::abs(previous[j] - next[j]) < 1e-7, "Trajectory is not C2 continuous");
    }
  }
  for (unsigned d = 1; d <= 2; ++d) {
    const auto first = segments.front().value(0., d), last = segments.back().value(1., d);
    for (unsigned j = 0; j < 7; ++j)
      require(std::abs(first[j]) < 1e-8 && std::abs(last[j]) < 1e-8, "Endpoints must have zero velocity and acceleration");
  }
  Joints uncertainty{};
  for (unsigned j = 0; j < 7; ++j)
    uncertainty[j] = tracking + start_budget + .5 * geometry_dt * velocity[j];
  const size_t count = static_cast<size_t>(std::ceil(duration() / geometry_dt));
  double minimum = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i <= count; ++i) {
    const double t = duration() * i / count;
    const double clearance = floor_clearance(value(t), uncertainty);
    require(std::isfinite(clearance) && clearance >= floor_margin, "Floor clearance bound failed at t=" + std::to_string(t));
    minimum = std::min(minimum, clearance);
  }
  std::cout << "Validated " << segments.size() << " spline segments; duration=" << duration()
            << " s; floor clearance lower bound=" << minimum << " m\n";
}

Joints start_offset(const Plan& p, const Joints& desired, const Joints& measured,
                    const Joints& velocity, const Joints& desired_velocity,
                    const Joints& desired_acceleration) {
  const auto first = p.value(0);
  Joints offset{};
  for (unsigned j = 0; j < 7; ++j) {
    require(std::isfinite(desired[j]) && std::isfinite(measured[j]) && std::isfinite(velocity[j]) &&
            std::isfinite(desired_velocity[j]) && std::isfinite(desired_acceleration[j]), "Invalid live state");
    offset[j] = desired[j] - first[j];
    require(std::abs(offset[j]) <= p.start_budget, "Robot moved since snapshot; regenerate the approach");
    require(std::abs(measured[j] - desired[j]) <= p.tracking, "Initial tracking error too large");
    require(std::abs(velocity[j]) < .01 && std::abs(desired_velocity[j]) < 1e-6 &&
            std::abs(desired_acceleration[j]) < 1e-5, "Robot must be stationary");
  }
  return offset;
}
}  // namespace panda_replay

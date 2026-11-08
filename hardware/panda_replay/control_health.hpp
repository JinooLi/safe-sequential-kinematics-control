#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace panda_replay {
// Policy for this replayer, not a replacement for the robot's communication guard.
// The strict default requires >=99% after 100 ms and periods <=2 ms. The explicit
// warn policy records those conditions without aborting; invalid feedback still
// aborts. Both policies hold the start pose for 500 ms before playback.
class CommunicationGate {
 public:
  enum class Policy { strict, warn };
  enum class Status { waiting, ready, invalid, poor_reception };
  explicit CommunicationGate(Policy policy = Policy::strict) noexcept : policy_(policy) {}
  Status update(std::uint64_t period_ms, double success_rate) noexcept {
    if ((samples_ && period_ms == 0) ||
        (policy_ == Policy::strict && period_ms > 2) ||
        !std::isfinite(success_rate) || success_rate < 0 || success_rate > 1 ||
        period_ms > std::numeric_limits<std::uint64_t>::max() - elapsed_ms_)
      return Status::invalid;
    ++samples_;
    elapsed_ms_ += period_ms;
    if (period_ms > 2) ++gaps_over_2ms_;
    if (elapsed_ms_ >= 100) {
      if (success_rate < min_success_) min_success_ = success_rate;
      if (success_rate < .99 - 1e-12) {
        ++poor_reception_samples_;
        if (policy_ == Policy::strict) return Status::poor_reception;
      }
    }
    return elapsed_ms_ >= 500 ? Status::ready : Status::waiting;
  }
  Policy policy() const noexcept { return policy_; }
  std::uint64_t samples() const noexcept { return samples_; }
  std::uint64_t elapsed_ms() const noexcept { return elapsed_ms_; }
  std::uint64_t gaps_over_2ms() const noexcept { return gaps_over_2ms_; }
  std::uint64_t poor_reception_samples() const noexcept { return poor_reception_samples_; }
  double min_success() const noexcept { return min_success_; }
 private:
  Policy policy_;
  std::uint64_t elapsed_ms_{};
  std::uint64_t samples_{};
  std::uint64_t gaps_over_2ms_{}, poor_reception_samples_{};
  double min_success_{1.};
};

// Compares the last raw position request with the robot's desired position.
// This is an application guard, separate from measured tracking and geometry.
// Existing lab controllers do not impose this 0.001 rad cancellation threshold.
class DesiredStateGate {
 public:
  enum class Policy { strict, warn };
  enum class Status { ok, invalid, deviation };
  explicit DesiredStateGate(Policy policy = Policy::strict) noexcept : policy_(policy) {}
  Status update(const std::array<double, 7>& reference,
                const std::array<double, 7>& desired) noexcept {
    for (unsigned j = 0; j < 7; ++j)
      if (!std::isfinite(reference[j]) || !std::isfinite(desired[j])) return Status::invalid;
    bool exceeded = false;
    for (unsigned j = 0; j < 7; ++j) {
      const double error = std::abs(desired[j] - reference[j]);
      if (!std::isfinite(error)) return Status::invalid;
      exceeded = exceeded || error > .001;
      if (error > max_deviation_) {
        max_deviation_ = error;
        max_joint_ = j + 1;
      }
    }
    if (exceeded) ++exceeded_samples_;
    return exceeded && policy_ == Policy::strict ? Status::deviation : Status::ok;
  }
  Policy policy() const noexcept { return policy_; }
  std::uint64_t exceeded_samples() const noexcept { return exceeded_samples_; }
  double max_deviation() const noexcept { return max_deviation_; }
  unsigned max_joint() const noexcept { return max_joint_; }
 private:
  Policy policy_;
  std::uint64_t exceeded_samples_{};
  double max_deviation_{};
  unsigned max_joint_{};
};
}  // namespace panda_replay

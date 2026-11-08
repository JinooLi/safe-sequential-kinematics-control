#pragma once
#include "trajectory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace panda_replay {
// A bounded trace: no file I/O, printing, or allocation in the callback.
// Completed callbacks stay in chronological order when dumped after control exits.
class TimingTrace {
 public:
  struct Sample {
    std::size_t index{};
    double robot_time_ms{}, period_ms{}, plan_time_s{}, success_rate{}, work_us{};
    bool has_feedback{};
    Joints reference{}, measured{}, desired{};
    double host_interval_ms{};
  };
  static constexpr std::size_t capacity = 1024;
  using Clock = std::chrono::steady_clock;
  class Scope {
   public:
    Scope(TimingTrace& trace, double robot_time_ms, double period_ms,
          double plan_time, double success_rate) noexcept
        : trace_(trace), start_(Clock::now()) {
      sample_ = {trace_.count_, robot_time_ms, period_ms, plan_time, success_rate, 0.};
      if (trace_.host_started_)
        sample_.host_interval_ms = std::chrono::duration<double, std::milli>(start_ - trace_.last_start_).count();
      trace_.last_start_ = start_;
      trace_.host_started_ = true;
    }
    ~Scope() noexcept {
      sample_.work_us = std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
      trace_.record(sample_);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void feedback(const Joints& reference, const Joints& measured, const Joints& desired) noexcept {
      sample_.has_feedback = true;
      sample_.reference = reference;
      sample_.measured = measured;
      sample_.desired = desired;
    }
   private:
    TimingTrace& trace_;
    Clock::time_point start_;
    Sample sample_;
  };

  void record(const Sample& sample) noexcept {
    samples_[count_ % capacity] = sample;
    ++count_;
    if (sample.period_ms > max_gap_.period_ms) max_gap_ = sample;
    max_work_us_ = std::max(max_work_us_, sample.work_us);
    max_host_gap_ms_ = std::max(max_host_gap_ms_, sample.host_interval_ms);
    if (sample.host_interval_ms > 1.5) ++host_over_1_5ms_;
    if (sample.period_ms > 1.) ++over_1ms_;
    if (sample.period_ms > 2.) ++over_2ms_;
    if (sample.index >= 100) min_success_ = std::min(min_success_, sample.success_rate);
    if (sample.has_feedback) for (unsigned j = 0; j < 7; ++j) {
      max_tracking_ = std::max(max_tracking_, std::abs(sample.measured[j] - sample.reference[j]));
      const double deviation = std::abs(sample.desired[j] - sample.reference[j]);
      if (deviation > max_desired_deviation_) {
        max_desired_deviation_ = deviation;
        deviation_joint_ = j + 1;
      }
    }
  }
  std::size_t size() const noexcept { return count_; }
  Sample at(std::size_t index) const noexcept { return samples_[index % capacity]; }
  std::size_t over_2ms() const noexcept { return over_2ms_; }
  void dump(const char* mode) const {
    if (!count_) return;
    std::cerr << std::setprecision(6) << "Timing (" << mode << "): callbacks=" << count_
              << ", max_gap_ms=" << max_gap_.period_ms
              << " at callback=" << max_gap_.index << " plan_t=" << max_gap_.plan_time_s
              << " s, gaps_over_1ms=" << over_1ms_ << ", gaps_over_2ms=" << over_2ms_
              << ", max_callback_work_us=" << max_work_us_
              << ", max_host_gap_ms=" << max_host_gap_ms_
              << ", host_gaps_over_1_5ms=" << host_over_1_5ms_ << '\n';
    std::cerr << "Control feedback: min_success_rate_after_100_callbacks=" << min_success_
              << ", max_measured_deviation_rad=" << max_tracking_
              << ", max_desired_deviation_rad=" << max_desired_deviation_
              << ", desired_deviation_joint=" << deviation_joint_ << '\n';
    const auto id = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string path = "panda_replay_timing_" + std::to_string(id) + ".csv";
    std::ofstream out(path);
    if (!out) {
      std::cerr << "Cannot write timing trace in the current directory\n";
      return;
    }
    out << "callback,robot_time_ms,period_ms,plan_time_s,control_command_success_rate,callback_work_us,has_feedback";
    for (const char* name : {"reference", "measured", "desired"})
      for (unsigned j = 1; j <= 7; ++j) out << ',' << name << '_' << j;
    out << ",host_interval_ms\n";
    out << std::setprecision(17);
    const std::size_t first = count_ > capacity ? count_ - capacity : 0;
    for (std::size_t i = first; i < count_; ++i) {
      const auto& s = samples_[i % capacity];
      out << s.index << ',' << s.robot_time_ms << ',' << s.period_ms << ',' << s.plan_time_s
          << ',' << s.success_rate << ',' << s.work_us << ',' << s.has_feedback;
      for (const auto* q : {&s.reference, &s.measured, &s.desired})
        for (double x : *q) out << ',' << x;
      out << ',' << s.host_interval_ms << '\n';
    }
    std::cerr << "Last " << count_ - first << " callback samples: " << path << '\n';
  }
 private:
  std::array<Sample, capacity> samples_{};
  Sample max_gap_{};
  std::size_t count_{}, over_1ms_{}, over_2ms_{};
  double max_work_us_{};
  double min_success_{1.}, max_tracking_{}, max_desired_deviation_{};
  unsigned deviation_joint_{};
  Clock::time_point last_start_{};
  bool host_started_{};
  double max_host_gap_ms_{};
  std::size_t host_over_1_5ms_{};
};
}  // namespace panda_replay

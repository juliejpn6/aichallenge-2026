// Copyright 2024 TIER IV, Inc.
// Copyright 2026 Taiki Tanaka
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_vehicle_msgs/msg/gear_command.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{

struct Point2D
{
  double x;
  double y;
};

std::vector<Point2D> load_raceline(const std::string & path, rclcpp::Logger logger)
{
  std::vector<Point2D> pts;
  if (path.empty()) {
    return pts;
  }
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    RCLCPP_WARN(logger, "Cannot open heading CSV: %s (raceline yaw disabled)", path.c_str());
    return pts;
  }
  std::string line;
  std::getline(ifs, line);  // skip header
  while (std::getline(ifs, line)) {
    try {
      std::istringstream ss(line);
      std::string tok;
      std::getline(ss, tok, ',');
      const double x = std::stod(tok);
      std::getline(ss, tok, ',');
      const double y = std::stod(tok);
      if (std::isfinite(x) && std::isfinite(y)) {
        pts.push_back({x, y});
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger, "Skipping invalid CSV line: %s", e.what());
    }
  }
  return pts;
}

size_t find_closest(const std::vector<Point2D> & pts, double qx, double qy)
{
  size_t best = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < pts.size(); ++i) {
    const double dx = pts[i].x - qx;
    const double dy = pts[i].y - qy;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }
  return best;
}

std::optional<double> compute_yaw(const std::vector<Point2D> & pts, size_t idx)
{
  constexpr double kMinSegLen2 = 1.0e-6;
  for (size_t i = idx; i + 1 < pts.size(); ++i) {
    const double dx = pts[i + 1].x - pts[i].x;
    const double dy = pts[i + 1].y - pts[i].y;
    if (dx * dx + dy * dy > kMinSegLen2) {
      return std::atan2(dy, dx);
    }
  }
  for (size_t i = idx; i > 0; --i) {
    const double dx = pts[i].x - pts[i - 1].x;
    const double dy = pts[i].y - pts[i - 1].y;
    if (dx * dx + dy * dy > kMinSegLen2) {
      return std::atan2(dy, dx);
    }
  }
  return std::nullopt;
}

// センシング切り分け計装A(2026-07-19、118節続報): GNSS生値の高周波ジッタを
// センサ由来ノイズの直接推定値として求める。既知の軌道モデルを使わず、直近窓の
// 点群に全最小二乗(直交回帰)で直線をあてはめ、進行方向に直交する成分の残差RMSを
// 返す(wp127-129衝突事象の事後解析「手法1」と同一の数式をライブ計装化)。
// 2x2共分散行列の固有ベクトル(主成分方向)を閉形式で求める(Eigen等の依存を増やさない)。
struct TimedPoint2D { double x; double y; double t; };

double line_fit_perp_residual_rms(const std::deque<TimedPoint2D> & pts)
{
  const size_t n = pts.size();
  if (n < 4) {
    return -1.0;
  }
  double mx = 0.0, my = 0.0;
  for (const auto & p : pts) { mx += p.x; my += p.y; }
  mx /= static_cast<double>(n);
  my /= static_cast<double>(n);
  double cxx = 0.0, cyy = 0.0, cxy = 0.0;
  for (const auto & p : pts) {
    const double dx = p.x - mx;
    const double dy = p.y - my;
    cxx += dx * dx;
    cyy += dy * dy;
    cxy += dx * dy;
  }
  cxx /= static_cast<double>(n);
  cyy /= static_cast<double>(n);
  cxy /= static_cast<double>(n);
  const double theta = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);  // 主成分(進行方向)の角度
  const double nx = -std::sin(theta);
  const double ny = std::cos(theta);                            // 直交(法線)方向
  double sum_sq = 0.0;
  for (const auto & p : pts) {
    const double dx = p.x - mx;
    const double dy = p.y - my;
    const double r = nx * dx + ny * dy;
    sum_sq += r * r;
  }
  return std::sqrt(sum_sq / static_cast<double>(n));
}

double stddev_scalar(const std::deque<double> & vals)
{
  const size_t n = vals.size();
  if (n < 4) {
    return -1.0;
  }
  double mean = 0.0;
  for (double v : vals) { mean += v; }
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (double v : vals) {
    const double d = v - mean;
    var += d * d;
  }
  var /= static_cast<double>(n);
  return std::sqrt(var);
}

}  // namespace

class ImuGnssPoser : public rclcpp::Node
{
public:
  ImuGnssPoser() : Node("imu_gnss_poser")
  {
    // Parameters
    declare_parameter("heading_csv_path", std::string(""));
    declare_parameter("initial_pose_service", std::string("/set_initial_pose"));
    declare_parameter("marker_topic", std::string("/heading_pose_initializer/raceline_markers"));
    declare_parameter("marker_publish_rate", 0.1);
    declare_parameter("arrow_interval", 2);
    declare_parameter("arrow_length", 1.0);

    // GNSS measurement covariance
    declare_parameter("gnss_covariance.good_threshold", 0.1);
    declare_parameter("gnss_covariance.good_value", 0.1);
    declare_parameter("gnss_covariance.moderate_threshold", 0.5);
    declare_parameter("gnss_covariance.moderate_value", 0.25);
    declare_parameter("gnss_covariance.poor_value", 100.0);
    declare_parameter("gnss_covariance.roll", 100000.0);
    declare_parameter("gnss_covariance.pitch", 100000.0);
    declare_parameter("gnss_covariance.yaw", 100000.0);

    // Initial pose covariance (for /set_initial_pose and first initial_pose3d)
    declare_parameter("initial_pose_covariance.x", 0.25);
    declare_parameter("initial_pose_covariance.y", 0.25);
    declare_parameter("initial_pose_covariance.yaw", 0.5);

    // GNSS-track heading injection (走行中、GNSS位置履歴から進行方位を求め EKF へ与える)
    //   方位は通常 gyro 積分の dead-reckoning のみで絶対補正が無く約-1.5°のバイアスが残る。
    //   GNSS軌跡の絶対方位を疎結合フュージョン(yaw cov 有限化)してバイアスを除去する。
    declare_parameter("gnss_heading.enable", true);
    declare_parameter("gnss_heading.track_dist", 0.5);     // [m] 進行方位を求める後方距離(長いほど低ノイズ・コーナーで遅れ大)
    declare_parameter("gnss_heading.max_dt", 1.0);         // [s] 後方点がこれより古い=低速/停止 → 注入せず(従来cov=100000へ退避)
    declare_parameter("gnss_heading.yaw_covariance", 0.2); // [rad^2] 注入方位の観測分散(保守的=大きめ。gyroが速い動特性, GNSSがバイアス補正)

    // センシング切り分け計装A(2026-07-19、118節続報): GNSS/IMU生値の高周波ノイズを
    //   直接推定し、[GNSS-NOISE]/[IMU-NOISE]としてログする。EKF平滑化の評価(B、
    //   mpc_controller.py側の既存[LOC-XCHECK]と時刻突合で比較)の分母として使う。
    declare_parameter("sensing_noise.window_s", 1.0);      // [s] ノイズ推定に使う直近窓の長さ(GNSS/IMU共通)
    declare_parameter("sensing_noise.log_interval_s", 1.0); // [s] ログ出力の間引き間隔
    declare_parameter("sensing_noise.min_disp_m", 0.3);     // [m] GNSS直線あてはめが意味を持つ最小窓内変位(停止時の除外)

    arrow_interval_ = get_parameter("arrow_interval").as_int();
    arrow_length_ = get_parameter("arrow_length").as_double();

    gnss_cov_good_thresh_ = get_parameter("gnss_covariance.good_threshold").as_double();
    gnss_cov_good_ = get_parameter("gnss_covariance.good_value").as_double();
    gnss_cov_mod_thresh_ = get_parameter("gnss_covariance.moderate_threshold").as_double();
    gnss_cov_mod_ = get_parameter("gnss_covariance.moderate_value").as_double();
    gnss_cov_poor_ = get_parameter("gnss_covariance.poor_value").as_double();
    gnss_cov_roll_ = get_parameter("gnss_covariance.roll").as_double();
    gnss_cov_pitch_ = get_parameter("gnss_covariance.pitch").as_double();
    gnss_cov_yaw_ = get_parameter("gnss_covariance.yaw").as_double();

    init_cov_x_ = get_parameter("initial_pose_covariance.x").as_double();
    init_cov_y_ = get_parameter("initial_pose_covariance.y").as_double();
    init_cov_yaw_ = get_parameter("initial_pose_covariance.yaw").as_double();

    gnss_heading_enable_ = get_parameter("gnss_heading.enable").as_bool();
    gnss_heading_track_dist_ = get_parameter("gnss_heading.track_dist").as_double();
    gnss_heading_max_dt_ = get_parameter("gnss_heading.max_dt").as_double();
    gnss_heading_yaw_cov_ = get_parameter("gnss_heading.yaw_covariance").as_double();

    noise_window_s_ = get_parameter("sensing_noise.window_s").as_double();
    noise_log_interval_s_ = get_parameter("sensing_noise.log_interval_s").as_double();
    noise_min_disp_m_ = get_parameter("sensing_noise.min_disp_m").as_double();

    // Load raceline
    const auto csv_path = get_parameter("heading_csv_path").as_string();
    raceline_ = load_raceline(csv_path, get_logger());
    has_raceline_ = raceline_.size() >= 2;
    if (has_raceline_) {
      RCLCPP_INFO(
        get_logger(), "Loaded %zu heading-reference points from %s",
        raceline_.size(), csv_path.c_str());
    }

    // QoS
    const auto rv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    const auto rt_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    // Publishers
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/imu_gnss_poser/pose_with_covariance", rv_qos);
    pub_initial_pose_3d_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/localization/initial_pose3d", rt_qos);

    // Subscriptions
    sub_gnss_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/sensing/gnss/pose_with_covariance", rv_qos,
      std::bind(&ImuGnssPoser::gnss_callback, this, std::placeholders::_1));
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      "/sensing/imu/imu_raw", rv_qos,
      std::bind(&ImuGnssPoser::imu_callback, this, std::placeholders::_1));
    // スタック検知バック(2026-07-10): 後退中はGNSS軌跡ベースのyaw推定(進行方向=車首方向前提)が
    //   180°反転して誤る。mpc_controller側が既に送っているgear_cmdを流用し、REVERSE中は
    //   apply_gnss_track_heading()での方位注入を止める(新規トピック追加を避けシンプルに)。
    sub_gear_ = create_subscription<autoware_auto_vehicle_msgs::msg::GearCommand>(
      "/control/command/gear_cmd", rv_qos,
      std::bind(&ImuGnssPoser::gear_callback, this, std::placeholders::_1));

    // EKF trigger client
    ekf_trigger_client_ = create_client<std_srvs::srv::SetBool>("/localization/trigger_node");

    // /set_initial_pose service
    const auto svc_name = get_parameter("initial_pose_service").as_string();
    service_ = create_service<std_srvs::srv::Trigger>(
      svc_name,
      std::bind(
        &ImuGnssPoser::on_set_initial_pose, this,
        std::placeholders::_1, std::placeholders::_2));

    // Raceline markers
    if (has_raceline_) {
      rclcpp::QoS mq(1);
      mq.reliable().transient_local();
      marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        get_parameter("marker_topic").as_string(), mq);
      markers_ = build_markers();
      marker_pub_->publish(markers_);
      const double rate = get_parameter("marker_publish_rate").as_double();
      if (rate > 0.0) {
        marker_timer_ = create_wall_timer(
          std::chrono::duration<double>(1.0 / rate),
          [this]() { marker_pub_->publish(markers_); });
      }
    }

    RCLCPP_INFO(
      get_logger(), "imu_gnss_poser ready (raceline=%s, service=%s)",
      has_raceline_ ? "yes" : "no", svc_name.c_str());
  }

private:
  // ── GNSS callback ──────────────────────────────────────────

  void gnss_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    adjust_covariance(*msg);
    apply_imu_orientation_fallback(*msg);
    apply_gnss_track_heading(*msg);  // GNSS軌跡から進行方位を与えヨーバイアスを補正(可能時のみ)

    // Publish fused pose for EKF measurement input (GNSS-track yaw if available, else IMU yaw)
    pub_pose_->publish(*msg);

    // Store latest for /set_initial_pose service
    {
      std::lock_guard<std::mutex> lk(gnss_mutex_);
      last_gnss_ = msg;
    }

    // センシング切り分け計装A(2026-07-19、118節続報): GNSS生値ノイズの計測は
    //   gnss_heading機能の有効/無効・後退中かどうかに関わらず常に行う(診断用途)。
    update_and_maybe_log_sensing_noise(
      rclcpp::Time(msg->header.stamp).seconds(),
      msg->pose.pose.position.x, msg->pose.pose.position.y);

    // Publish initial_pose3d until EKF is triggered.
    // 起動時の初期姿勢には GNSS/IMU の向き(静止時はヨー不定)でなくレースライン向きを与える。
    // これが無いと EKF が yaw=0/cov=100000 から約3秒かけて収束し、発進直後に誤操舵する。
    // ※EKF計測入力(pub_pose_)は GNSS/IMU 向きのまま。初期姿勢コピーにのみ向きを上書き。
    // ※この経路は EKF トリガ前(!ekf_triggered_)のみ＝起動時1回限り。レース中・衝突後の挙動には不関与。
    if (!ekf_triggered_) {
      auto init_msg = *msg;
      try_apply_raceline_yaw(init_msg);  // 失敗時(raceline未読込/位置不正)は元のまま=従来挙動
      pub_initial_pose_3d_->publish(init_msg);
      if (!initial_pose_published_) {
        RCLCPP_INFO(get_logger(), "Publishing initial_pose3d (raceline yaw applied)");
        initial_pose_published_ = true;
      }
      try_trigger_ekf();
    }
  }

  void adjust_covariance(geometry_msgs::msg::PoseWithCovarianceStamped & msg) const
  {
    auto adj = [this](double v) -> double {
      if (v <= gnss_cov_good_thresh_) return gnss_cov_good_;
      if (v <= gnss_cov_mod_thresh_) return gnss_cov_mod_;
      return gnss_cov_poor_;
    };
    msg.pose.covariance[7 * 0] = adj(msg.pose.covariance[7 * 0]);
    msg.pose.covariance[7 * 1] = adj(msg.pose.covariance[7 * 1]);
    msg.pose.covariance[7 * 2] = adj(msg.pose.covariance[7 * 2]);
    msg.pose.covariance[7 * 3] = gnss_cov_roll_;
    msg.pose.covariance[7 * 4] = gnss_cov_pitch_;
    msg.pose.covariance[7 * 5] = gnss_cov_yaw_;
  }

  bool try_apply_raceline_yaw(geometry_msgs::msg::PoseWithCovarianceStamped & msg) const
  {
    if (!has_raceline_) {
      return false;
    }
    const auto & pos = msg.pose.pose.position;
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y)) {
      return false;
    }
    const auto idx = find_closest(raceline_, pos.x, pos.y);
    const auto yaw = compute_yaw(raceline_, idx);
    if (!yaw.has_value()) {
      return false;
    }
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = std::sin(*yaw * 0.5);
    msg.pose.pose.orientation.w = std::cos(*yaw * 0.5);
    msg.pose.covariance[7 * 5] = init_cov_yaw_;
    return true;
  }

  void apply_imu_orientation_fallback(geometry_msgs::msg::PoseWithCovarianceStamped & msg) const
  {
    const auto & o = msg.pose.pose.orientation;
    if (std::isnan(o.x) || std::isnan(o.y) || std::isnan(o.z) || std::isnan(o.w) ||
      (o.x == 0 && o.y == 0 && o.z == 0 && o.w == 0))
    {
      msg.pose.pose.orientation = imu_msg_.orientation;
    }
  }

  // ── GNSS-track heading injection ───────────────────────────
  // GNSS位置履歴から進行方位(後方 track_dist 離れた点との atan2)を求め、
  // EKFへ渡す pose の方位として与える(yaw cov を有限化)。これにより
  // gyro 積分だけでは補正されない方位バイアス(約-1.5°)を絶対方位で除去する。
  // 低速/微小移動時(後方点が古い/見つからない)は注入せず、従来挙動(cov=100000, IMU方位)へ退避。
  void apply_gnss_track_heading(geometry_msgs::msg::PoseWithCovarianceStamped & msg)
  {
    if (!gnss_heading_enable_) {
      return;
    }
    // 2026-07-10: 後退中(gear=REVERSE)は移動方向≠車首方向のため、この推定は使わない
    // (従来挙動=cov=100000, IMU方位ベースへ退避。位置(x,y)自体は後退中も正しく使う)。
    if (is_reversing_) {
      return;
    }
    const auto & pos = msg.pose.pose.position;
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y)) {
      return;
    }
    const double t_now = rclcpp::Time(msg.header.stamp).seconds();
    gnss_hist_.push_back({pos.x, pos.y, t_now});
    while (gnss_hist_.size() > 200) {
      gnss_hist_.pop_front();
    }
    const double d2_thresh = gnss_heading_track_dist_ * gnss_heading_track_dist_;
    // 直近から遡り、track_dist 以上離れた最初の過去点で方位を計算
    for (auto it = gnss_hist_.rbegin(); it != gnss_hist_.rend(); ++it) {
      const double dx = pos.x - it->x;
      const double dy = pos.y - it->y;
      if (dx * dx + dy * dy >= d2_thresh) {
        // 速度(時間)ゲート: 後方点が古すぎる=低速/停止 → 方位不定なので注入しない
        if (t_now - it->t > gnss_heading_max_dt_) {
          return;
        }
        const double yaw = std::atan2(dy, dx);
        msg.pose.pose.orientation.x = 0.0;
        msg.pose.pose.orientation.y = 0.0;
        msg.pose.pose.orientation.z = std::sin(yaw * 0.5);
        msg.pose.pose.orientation.w = std::cos(yaw * 0.5);
        msg.pose.covariance[7 * 5] = gnss_heading_yaw_cov_;  // yaw 観測を EKF に使わせる
        return;
      }
    }
    // 十分な移動がなければ従来挙動(cov=100000)のまま
  }

  // ── IMU callback ───────────────────────────────────────────

  void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg)
  {
    imu_msg_ = *msg;
    // センシング切り分け計装A(2026-07-19、118節続報): [IMU-NOISE]用にgyro wzの
    //   履歴を保持する(実際のログ出力・間引きはGNSSコールバック側で行う、新規タイマ不要)。
    noise_wz_val_.push_back(msg->angular_velocity.z);
    noise_wz_t_.push_back(rclcpp::Time(msg->header.stamp).seconds());
  }

  // ── センシング切り分け計装A(2026-07-19、118節続報) ─────────
  // [GNSS-NOISE]/[IMU-NOISE]: 既知の軌道モデルを使わず、直近窓の生値そのものから
  // センサ単体のノイズを直接推定する。EKF平滑化の評価(B)は、これらの値と既存
  // [LOC-XCHECK](mpc_controller.py)のekf_ey-gnss_ey差を時刻突合して行う(本ノード
  // 側では計算しない=非冗長)。
  void update_and_maybe_log_sensing_noise(double t_now, double gx, double gy)
  {
    noise_gnss_hist_.push_back({gx, gy, t_now});
    while (!noise_gnss_hist_.empty() &&
      t_now - noise_gnss_hist_.front().t > noise_window_s_)
    {
      noise_gnss_hist_.pop_front();
    }
    while (!noise_wz_t_.empty() && t_now - noise_wz_t_.front() > noise_window_s_) {
      noise_wz_t_.pop_front();
      noise_wz_val_.pop_front();
    }

    if (noise_last_log_t_ >= 0.0 && t_now - noise_last_log_t_ < noise_log_interval_s_) {
      return;
    }
    noise_last_log_t_ = t_now;

    if (noise_gnss_hist_.size() >= 4) {
      const double dispx = noise_gnss_hist_.back().x - noise_gnss_hist_.front().x;
      const double dispy = noise_gnss_hist_.back().y - noise_gnss_hist_.front().y;
      const double disp = std::hypot(dispx, dispy);
      if (disp >= noise_min_disp_m_) {
        const double rms = line_fit_perp_residual_rms(noise_gnss_hist_);
        RCLCPP_INFO(
          get_logger(), "[GNSS-NOISE] rms=%.4f n=%zu window=%.1f disp=%.2f",
          rms, noise_gnss_hist_.size(), noise_window_s_, disp);
      }
    }
    if (noise_wz_val_.size() >= 4) {
      const double wz_std = stddev_scalar(noise_wz_val_);
      RCLCPP_INFO(
        get_logger(), "[IMU-NOISE] wz_std=%.5f n=%zu window=%.1f",
        wz_std, noise_wz_val_.size(), noise_window_s_);
    }
  }

  // ── Gear callback(2026-07-10、スタック検知バック対応) ──────
  void gear_callback(autoware_auto_vehicle_msgs::msg::GearCommand::SharedPtr msg)
  {
    is_reversing_ = (msg->command == autoware_auto_vehicle_msgs::msg::GearCommand::REVERSE);
  }

  // ── EKF trigger ────────────────────────────────────────────

  void try_trigger_ekf()
  {
    if (!ekf_trigger_client_->service_is_ready()) {
      return;  // will retry on next GNSS callback
    }
    auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
    req->data = true;
    ekf_trigger_client_->async_send_request(
      req,
      [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
        const auto resp = future.get();
        RCLCPP_INFO(
          get_logger(), "EKF trigger: success=%s", resp->success ? "true" : "false");
      });
    ekf_triggered_ = true;
    RCLCPP_INFO(get_logger(), "Called EKF trigger");
  }

  // ── /set_initial_pose service ──────────────────────────────

  void on_set_initial_pose(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!has_raceline_) {
      response->success = false;
      response->message = "heading CSV not loaded";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr gnss;
    {
      std::lock_guard<std::mutex> lk(gnss_mutex_);
      gnss = last_gnss_;
    }
    if (!gnss) {
      response->success = false;
      response->message = "no GNSS data received yet";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    const auto & pos = gnss->pose.pose.position;
    if (!std::isfinite(pos.x) || !std::isfinite(pos.y)) {
      response->success = false;
      response->message = "GNSS position is invalid (NaN/Inf)";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    const auto idx = find_closest(raceline_, pos.x, pos.y);
    const auto yaw = compute_yaw(raceline_, idx);
    if (!yaw.has_value()) {
      response->success = false;
      response->message = "cannot compute yaw from heading reference";
      RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
      return;
    }

    geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
    pose_msg.header.stamp = this->now();
    pose_msg.header.frame_id = gnss->header.frame_id;
    pose_msg.pose.pose.position = gnss->pose.pose.position;
    pose_msg.pose.pose.orientation.z = std::sin(*yaw * 0.5);
    pose_msg.pose.pose.orientation.w = std::cos(*yaw * 0.5);
    pose_msg.pose.covariance[7 * 0] = init_cov_x_;
    pose_msg.pose.covariance[7 * 1] = init_cov_y_;
    pose_msg.pose.covariance[7 * 5] = init_cov_yaw_;

    pub_initial_pose_3d_->publish(pose_msg);

    // Call trigger directly without resetting ekf_triggered_,
    // so gnss_callback won't re-publish initial_pose3d continuously.
    if (ekf_trigger_client_->service_is_ready()) {
      auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
      req->data = true;
      ekf_trigger_client_->async_send_request(req);
    }

    const double yaw_deg = *yaw * 180.0 / M_PI;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "published initial pose (yaw %.1f deg)", yaw_deg);
    response->success = true;
    response->message = buf;
    RCLCPP_INFO(get_logger(), "%s", buf);
  }

  // ── Raceline markers ──────────────────────────────────────

  visualization_msgs::msg::MarkerArray build_markers() const
  {
    visualization_msgs::msg::MarkerArray ma;
    if (!has_raceline_) {
      return ma;
    }
    const auto now = this->now();
    int arrow_id = 0;
    for (size_t i = 0; i + 1 < raceline_.size();
      i += static_cast<size_t>(arrow_interval_))
    {
      const auto yaw = compute_yaw(raceline_, i);
      if (!yaw.has_value()) {
        continue;
      }
      visualization_msgs::msg::Marker arrow;
      arrow.header.frame_id = "map";
      arrow.header.stamp = now;
      arrow.ns = "heading_arrows";
      arrow.id = arrow_id++;
      arrow.type = visualization_msgs::msg::Marker::ARROW;
      arrow.action = visualization_msgs::msg::Marker::ADD;

      geometry_msgs::msg::Point start;
      start.x = raceline_[i].x;
      start.y = raceline_[i].y;
      start.z = 0.5;
      geometry_msgs::msg::Point end;
      end.x = raceline_[i].x + arrow_length_ * std::cos(*yaw);
      end.y = raceline_[i].y + arrow_length_ * std::sin(*yaw);
      end.z = 0.5;
      arrow.points.push_back(start);
      arrow.points.push_back(end);

      arrow.scale.x = 0.25;
      arrow.scale.y = 0.3;
      arrow.scale.z = 0.2;
      arrow.color.r = 1.0f;
      arrow.color.g = 1.0f;
      arrow.color.b = 1.0f;
      arrow.color.a = 0.5f;
      ma.markers.push_back(arrow);
    }
    return ma;
  }

  // ── Members ────────────────────────────────────────────────

  // GNSS measurement covariance
  double gnss_cov_good_thresh_{0.1};
  double gnss_cov_good_{0.1};
  double gnss_cov_mod_thresh_{0.5};
  double gnss_cov_mod_{0.25};
  double gnss_cov_poor_{100.0};
  double gnss_cov_roll_{100000.0};
  double gnss_cov_pitch_{100000.0};
  double gnss_cov_yaw_{100000.0};

  // Initial pose covariance
  double init_cov_x_{0.25};
  double init_cov_y_{0.25};
  double init_cov_yaw_{0.5};

  // GNSS-track heading injection
  struct TimedPoint { double x; double y; double t; };
  bool gnss_heading_enable_{true};
  double gnss_heading_track_dist_{0.5};
  double gnss_heading_max_dt_{1.0};
  double gnss_heading_yaw_cov_{0.2};
  std::deque<TimedPoint> gnss_hist_;

  // センシング切り分け計装A(2026-07-19、118節続報): GNSS/IMU生ノイズ推定。
  //   gnss_hist_(gnss_heading専用、enable=false/後退中は更新されない)とは別に、
  //   常に更新される専用バッファを持つ(診断はheading注入の有効/無効に依存しない)。
  double noise_window_s_{1.0};
  double noise_log_interval_s_{1.0};
  double noise_min_disp_m_{0.3};
  std::deque<TimedPoint2D> noise_gnss_hist_;
  std::deque<double> noise_wz_val_;
  std::deque<double> noise_wz_t_;
  double noise_last_log_t_{-1.0};

  // Raceline
  std::vector<Point2D> raceline_;
  bool has_raceline_{false};
  int64_t arrow_interval_{2};
  double arrow_length_{1.0};

  // State
  bool initial_pose_published_{false};
  bool ekf_triggered_{false};
  std::mutex gnss_mutex_;
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr last_gnss_;
  sensor_msgs::msg::Imu imu_msg_;

  // ROS interfaces
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_initial_pose_3d_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_gnss_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<autoware_auto_vehicle_msgs::msg::GearCommand>::SharedPtr sub_gear_;
  bool is_reversing_{false};  // 2026-07-10: gear=REVERSE中はGNSS軌跡ベースyaw推定を止める
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr ekf_trigger_client_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr service_;
  rclcpp::TimerBase::SharedPtr marker_timer_;
  visualization_msgs::msg::MarkerArray markers_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImuGnssPoser>());
  rclcpp::shutdown();
  return 0;
}

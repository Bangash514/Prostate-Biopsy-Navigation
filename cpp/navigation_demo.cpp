// BEN 2026-11-01
// Change aim: C++ (Eigen + OpenCV) port of navigation_demo.py: point registration, pivot calibration, targeting error.
// File: navigation_demo.cpp
// SIMULATED data only (no patient images). Units: millimetres. Research/portfolio code, not a medical device.
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;

static const double kFleSigma = 0.5;      // fiducial localisation noise per axis (mm), an assumption
static const double kPoseSigma = 0.3;     // tool position noise per axis (mm), an assumption
static const double kRotSigmaDeg = 0.2;   // tool orientation noise (deg), an assumption
static const double kPi = 3.14159265358979323846;

static Matrix3d eulerToMatrix(double rxDeg, double ryDeg, double rzDeg)
{
    const double rx = rxDeg * kPi / 180, ry = ryDeg * kPi / 180, rz = rzDeg * kPi / 180;
    Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0, 0, std::cos(rx), -std::sin(rx), 0, std::sin(rx), std::cos(rx);
    Ry << std::cos(ry), 0, std::sin(ry), 0, 1, 0, -std::sin(ry), 0, std::cos(ry);
    Rz << std::cos(rz), -std::sin(rz), 0, std::sin(rz), std::cos(rz), 0, 0, 0, 1;
    return Rz * Ry * Rx;
}

static Matrix3d axisAngle(const Vector3d& axis, double angle)
{
    Matrix3d K;
    K << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(), axis.x(), 0;
    return Matrix3d::Identity() + std::sin(angle) * K + (1 - std::cos(angle)) * K * K;
}

struct Rigid { Matrix3d R; Vector3d t; };

static Rigid registerRigid(const MatrixXd& a, const MatrixXd& b)
{
    const Vector3d ca = a.rowwise().mean(), cb = b.rowwise().mean();
    const MatrixXd H = (a.colwise() - ca) * (b.colwise() - cb).transpose();
    Eigen::JacobiSVD<MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Matrix3d D = Matrix3d::Identity();
    D(2, 2) = ((V * U.transpose()).determinant() >= 0) ? 1.0 : -1.0;
    Rigid r;
    r.R = V * D * U.transpose();
    r.t = cb - r.R * ca;
    return r;
}

static double freRms(const Rigid& T, const MatrixXd& a, const MatrixXd& b)
{
    double s = 0;
    for (int i = 0; i < a.cols(); ++i) s += (T.R * a.col(i) + T.t - b.col(i)).squaredNorm();
    return std::sqrt(s / a.cols());
}

static double predictedTre(const MatrixXd& a, const Vector3d& target, double fleSigma)
{
    const int n = static_cast<int>(a.cols());
    const Vector3d ca = a.rowwise().mean();
    const MatrixXd q = a.colwise() - ca;
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(q * q.transpose());
    const Vector3d r = target - ca;
    double s = 0;
    for (int k = 0; k < 3; ++k) {
        const Vector3d v = es.eigenvectors().col(k);
        const double d2 = r.squaredNorm() - std::pow(r.dot(v), 2);
        double f2 = 0;
        for (int i = 0; i < n; ++i) f2 += q.col(i).squaredNorm() - std::pow(q.col(i).dot(v), 2);
        f2 /= n;
        s += d2 / f2;
    }
    const double fle2 = 3 * fleSigma * fleSigma;
    return std::sqrt(fle2 / n * (1 + s / 3));
}

static Vector3d pivotCalibration(const std::vector<Matrix3d>& rots, const std::vector<Vector3d>& pos)
{
    const int m = static_cast<int>(rots.size());
    MatrixXd A(3 * m, 6);
    Eigen::VectorXd b(3 * m);
    for (int i = 0; i < m; ++i) {
        A.block<3, 3>(3 * i, 0) = rots[i];
        A.block<3, 3>(3 * i, 3) = -Matrix3d::Identity();
        b.segment<3>(3 * i) = -pos[i];
    }
    const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    return x.head<3>();
}

struct Sim {
    std::mt19937_64 rng;
    explicit Sim(unsigned long long seed) : rng(seed) {}
    double gauss() { return std::normal_distribution<double>(0, 1)(rng); }
    double uni(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); }
    Vector3d gauss3() { return Vector3d(gauss(), gauss(), gauss()); }
};

static Vector3d calibrateOnce(Sim& sim, const Vector3d& tipTrue, const Vector3d& pivotTrue, int nPoses)
{
    std::vector<Matrix3d> rots;
    std::vector<Vector3d> pos;
    for (int i = 0; i < nPoses; ++i) {
        Vector3d axis = sim.gauss3();
        axis.normalize();
        const Matrix3d R = axisAngle(axis, sim.uni(5, 40) * kPi / 180);
        const Vector3d p = pivotTrue - R * tipTrue;
        Vector3d v = (kRotSigmaDeg * kPi / 180) * sim.gauss3();
        const double th = v.norm();
        const Matrix3d dR = (th < 1e-12) ? Matrix3d::Identity() : axisAngle(v / th, th);
        rots.push_back(dR * R);
        pos.push_back(p + kPoseSigma * sim.gauss3());
    }
    return pivotCalibration(rots, pos);
}

static std::string f(double v, int d)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(d) << v;
    return s.str();
}

struct Setup {
    MatrixXd fid;
    Vector3d target, tipTrue, pivotTrue, tTrue;
    Matrix3d RTrue;
};

static Setup makeSetup()
{
    Setup s;
    const double fid[6][3] = {{-55, -40, -25}, {50, -45, 20}, {58, 35, -22}, {-48, 42, 28}, {5, -5, 38}, {0, 50, -30}};
    s.fid.resize(3, 6);
    for (int i = 0; i < 6; ++i) s.fid.col(i) = Vector3d(fid[i][0], fid[i][1], fid[i][2]);
    s.target = Vector3d(8, -5, 12);
    s.tipTrue = Vector3d(0, 0, -150);
    s.pivotTrue = Vector3d(100, -30, 320);
    s.tTrue = Vector3d(120, -40, 300);
    s.RTrue = eulerToMatrix(15, -10, 25);
    return s;
}

static int checkFile(const std::string& path)
{
    std::vector<Vector3d> img, trk;
    Vector3d tgt = Vector3d::Zero();
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tag, v;
        std::getline(ss, tag, ',');
        double x[3];
        for (int i = 0; i < 3; ++i) { std::getline(ss, v, ','); x[i] = std::stod(v); }
        const Vector3d p(x[0], x[1], x[2]);
        if (tag == "img") img.push_back(p);
        else if (tag == "trk") trk.push_back(p);
        else if (tag == "target") tgt = p;
    }
    if (img.empty() || img.size() != trk.size()) { std::cerr << "bad check file\n"; return 1; }
    MatrixXd a(3, img.size()), b(3, img.size());
    for (size_t i = 0; i < img.size(); ++i) { a.col(i) = img[i]; b.col(i) = trk[i]; }
    const Rigid T = registerRigid(a, b);
    std::cout << "R:";
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) std::cout << " " << f(T.R(i, j), 5);
    std::cout << "\nt: " << f(T.t.x(), 5) << " " << f(T.t.y(), 5) << " " << f(T.t.z(), 5) << "\n";
    std::cout << "FRE: " << f(freRms(T, a, b), 5) << "\n";
    const Vector3d p = T.R * tgt + T.t;
    std::cout << "target in tracker space: " << f(p.x(), 5) << " " << f(p.y(), 5) << " " << f(p.z(), 5) << "\n";
    std::cout << "predicted RMS TRE: " << f(predictedTre(a, tgt, kFleSigma), 5) << "\n";
    return 0;
}

static void drawSlice(const std::string& path, double ex, double ey)
{
    const int size = 256;
    const double ppm = 4.0;
    cv::Mat a(size, size, CV_32F), b(size, size, CV_32F);
    cv::RNG rng(7);
    rng.fill(a, cv::RNG::NORMAL, 0, 1);
    rng.fill(b, cv::RNG::NORMAL, 0, 1);
    cv::GaussianBlur(a, a, cv::Size(0, 0), 1.2);
    cv::GaussianBlur(b, b, cv::Size(0, 0), 1.2);
    cv::Mat img(size, size, CV_32F);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            img.at<float>(y, x) = std::sqrt(a.at<float>(y, x) * a.at<float>(y, x) + b.at<float>(y, x) * b.at<float>(y, x));
    img /= cv::mean(img)[0];
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const double mx = (x - size / 2.0) / ppm, my = (y - size / 2.0) / ppm;
            const double gland = std::pow((mx + 3) / 24, 2) + std::pow((my - 2) / 18, 2);
            float v = img.at<float>(y, x);
            if (gland < 1) v *= 0.55f;
            if (gland > 0.9 && gland < 1.05) v *= 1.4f;
            if (mx * mx + my * my < 16.0) v *= 0.25f;
            img.at<float>(y, x) = v;
        }
    cv::Mat g8, col0, col;
    img.convertTo(g8, CV_8U, 110.0);
    cv::cvtColor(g8, col0, cv::COLOR_GRAY2BGR);
    cv::resize(col0, col, cv::Size(512, 512), 0, 0, cv::INTER_CUBIC);
    const double ppm2 = 8.0;
    const int half = 256;
    const cv::Point c(half, half);
    const cv::Point est(static_cast<int>(std::lround(half + ex * ppm2)), static_cast<int>(std::lround(half + ey * ppm2)));
    cv::line(col, cv::Point(c.x + 80, 0), est, cv::Scalar(255, 200, 0), 1, cv::LINE_AA);
    cv::circle(col, c, 32, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::drawMarker(col, est, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 1);
    cv::line(col, cv::Point(20, 492), cv::Point(20 + static_cast<int>(10 * ppm2), 492), cv::Scalar(255, 255, 255), 2);
    cv::putText(col, "10 mm", cv::Point(20, 478), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(col, "SIMULATED slice", cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(col, "green: true target   red: registered estimate", cv::Point(10, 48), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::imwrite(path, col);
}

static void monteCarlo(unsigned long long seed, int trials, const std::string& outDir)
{
    const Setup s = makeSetup();
    const int n = static_cast<int>(s.fid.cols());
    Sim sim(seed);
    const Vector3d targetTrk = s.RTrue * s.target + s.tTrue;
    const Matrix3d needleR = eulerToMatrix(0, 30, 0);
    double fre2 = 0, tre2 = 0, tipSum = 0;
    std::vector<double> total;
    struct Row { double tre, tip, tot, ex, ey, ez; };
    std::vector<Row> rows;
    for (int i = 0; i < trials; ++i) {
        MatrixXd b(3, n);
        for (int k = 0; k < n; ++k) b.col(k) = s.RTrue * s.fid.col(k) + s.tTrue + kFleSigma * sim.gauss3();
        const Rigid T = registerRigid(s.fid, b);
        const double fre = freRms(T, s.fid, b);
        fre2 += fre * fre;
        const Vector3d eReg = (T.R * s.target + T.t) - targetTrk;
        tre2 += eReg.squaredNorm();
        const Vector3d tipEst = calibrateOnce(sim, s.tipTrue, s.pivotTrue, 30);
        const Vector3d eTip = needleR * (s.tipTrue - tipEst);
        const double tipErr = (s.tipTrue - tipEst).norm();
        tipSum += tipErr;
        const double tot = (eReg + eTip).norm();
        total.push_back(tot);
        rows.push_back({eReg.norm(), tipErr, tot, eReg.x(), eReg.y(), eReg.z()});
    }
    std::vector<double> sorted = total;
    std::sort(sorted.begin(), sorted.end());
    double mean = 0;
    for (double v : total) mean += v;
    mean /= trials;
    const double median = (trials % 2) ? sorted[trials / 2] : 0.5 * (sorted[trials / 2 - 1] + sorted[trials / 2]);
    const double p95 = sorted[static_cast<size_t>(std::floor(0.95 * (trials - 1)))];
    auto within = [&](double mm) { int c = 0; for (double v : total) if (v <= mm) ++c; return 100.0 * c / trials; };

    std::ostringstream o;
    o << "fiducials: " << n << ", FLE sigma " << f(kFleSigma, 1) << " mm per axis, trials " << trials << "\n";
    o << "RMS FRE measured:  " << f(std::sqrt(fre2 / trials), 4) << " mm\n";
    o << "RMS FRE predicted: " << f(std::sqrt((1.0 - 2.0 / n) * 3 * kFleSigma * kFleSigma), 4) << " mm\n";
    o << "RMS TRE measured:  " << f(std::sqrt(tre2 / trials), 4) << " mm\n";
    o << "RMS TRE predicted: " << f(predictedTre(s.fid, s.target, kFleSigma), 4) << " mm\n";
    o << "mean tip calibration error: " << f(tipSum / trials, 4) << " mm\n";
    o << "targeting error mean: " << f(mean, 4) << " mm, median " << f(median, 4) << " mm, 95th percentile " << f(p95, 4) << " mm\n";
    o << "within 2 mm: " << f(within(2), 1) << " %, within 3 mm: " << f(within(3), 1) << " %, within 5 mm: " << f(within(5), 1) << " %\n";
    std::cout << o.str();
    std::ofstream(outDir + "/summary_cpp.txt") << o.str();
    std::ofstream csv(outDir + "/trials_cpp.csv");
    csv << "trial,tre_mm,tip_cal_error_mm,targeting_error_mm,ex,ey,ez\n";
    for (size_t i = 0; i < rows.size(); ++i)
        csv << i << "," << f(rows[i].tre, 5) << "," << f(rows[i].tip, 5) << "," << f(rows[i].tot, 5) << "," << f(rows[i].ex, 5)
            << "," << f(rows[i].ey, 5) << "," << f(rows[i].ez, 5) << "\n";
    std::vector<Row> byTre = rows;
    std::sort(byTre.begin(), byTre.end(), [](const Row& x, const Row& y) { return x.tre < y.tre; });
    const Row& m = byTre[byTre.size() / 2];
    drawSlice(outDir + "/slice_cpp.png", m.ex, m.ey);
}

int main(int argc, char** argv)
{
    std::string out = "results", check;
    unsigned long long seed = 1;
    int trials = 5000;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--out") out = argv[i + 1];
        else if (k == "--seed") seed = std::stoull(argv[i + 1]);
        else if (k == "--trials") trials = std::stoi(argv[i + 1]);
        else if (k == "--check") check = argv[i + 1];
    }
    if (!check.empty()) return checkFile(check);
    monteCarlo(seed, trials, out);
    return 0;
}

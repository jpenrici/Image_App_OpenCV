/**
 * @file imgtools.hpp
 * @brief High-level image comparison utilities based on OpenCV.
 *
 * This module provides a unified interface for loading images and performing:
 *   - Basic metadata comparison
 *   - Color-space analysis
 *   - Histogram similarity
 *   - Structural similarity (MSE, PSNR, SSIM)
 *   - Feature-based matching (ORB, AKAZE, SIFT)
 *
 * It is designed so that the GUI (Qt6/GTKmm) can display raw computation data:
 *   - Histograms
 *   - SSIM maps
 *   - Absolute difference maps
 *   - Keypoints, descriptors, match lines
 *   - RANSAC inliers and homography transforms
 *
 * All compute_* functions return raw data structs.
 * All compare_* functions return human-readable summaries.
 */

#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include <expected>
#include <filesystem>
#include <utility>

namespace imgtools {

/**
 * @brief Methods available for local feature detection and description.
 *
 * - ORB: Fast binary descriptors, excellent for real-time applications.
 * - AKAZE: Robust nonlinear scale-space; good balance of speed/stability.
 * - SIFT: Float descriptors; highest accuracy but slower.
 */
enum class FeatureMethod { ORB, AKAZE, SIFT };

/**
 * @brief Histogram comparison result.
 *
 * Stores:
 *   - Raw histograms (grayscale, normalized)
 *   - Individual similarity metrics
 *
 * These values are used both for textual analysis and for GUI visualization.
 */
struct HistogramResult {
    cv::Mat hist1;
    cv::Mat hist2;

    double correlation = 0.0;
    double chiSquare = 0.0;
    double intersection = 0.0;
    double bhattacharyya = 0.0;
    double kldiv = 0.0;
};

/**
 * @brief Structural comparison result (MSE/PSNR/SSIM).
 *
 * The GUI can display:
 *   - SSIM heatmap (ssimMap)
 *   - Absolute difference image (absDiff)
 */
struct StructuralResult {
    double mse = 0.0;
    double psnr = 0.0;
    double ssim = 0.0;

    cv::Mat ssimMap;  ///< SSIM map where values ∈ [0,1].
    cv::Mat absDiff;  ///< Absolute difference (grayscale).
};

/**
 * @brief Feature-based analysis result.
 *
 * This struct exposes all meaningful internal data for GUI visualization:
 *
 * Keypoints:
 *   - keypoints1, keypoints2
 *
 * Descriptors (ORB/AHAZE: binary; SIFT: float):
 *   - descriptors1, descriptors2
 *
 * Matching:
 *   - matches         → after Lowe ratio test
 *   - inliersMask     → mask returned by RANSAC
 *   - homography      → if enough matches found
 *
 * Metrics:
 *   - inlierRatio     → (#inliers / #matches)
 *   - meanDistance    → average descriptor distance
 *   - distanceVariance
 *   - meanRatio       → mean Lowe ratio
 */
struct FeatureResult {
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    cv::Mat descriptors1;
    cv::Mat descriptors2;

    std::vector<cv::DMatch> matches;
    std::vector<char> inliersMask;

    cv::Mat homography = cv::Mat();
    FeatureMethod method = FeatureMethod::AKAZE;

    double inlierRatio = 0.0;
    double meanDistance = 0.0;
    double distanceVariance = 0.0;
    double meanRatio = 0.0;
};

/**
 * @class ImageAnalyzer
 * @brief Unified image comparison engine.
 *
 * This class is designed for both console usage and GUI integration.
 * Its responsibilities:
 *
 * 1. Load images (color + grayscale)
 * 2. Perform comparison operations
 * 3. Provide raw data for visualization (via compute_* functions)
 * 4. Provide human-readable summaries (via compare_* functions)
 *
 * Example (GUI usage):
 *   ImageAnalyzer a(pathA, pathB);
 *   a.load_images();
 *   auto features = a.compute_features(FeatureMethod::ORB);
 *   // GUI will draw: keypoints, matches, inliers, homography overlay
 */
class ImageAnalyzer {
public:
    explicit ImageAnalyzer(std::string_view path1, std::string_view path2) noexcept;
    ~ImageAnalyzer() noexcept = default;

    ImageAnalyzer(const ImageAnalyzer&) = default;
    auto operator=(const ImageAnalyzer&) -> ImageAnalyzer& = default;

    ImageAnalyzer(ImageAnalyzer&&) noexcept = default;
    auto operator=(ImageAnalyzer&&) noexcept -> ImageAnalyzer& = default;

    /**
     * @brief Loads both images (color + grayscale).
     *
     * After this call:
     *   - image1_, image2_ contain full-color images.
     *   - grayscale1_, grayscale2_ contain cv::COLOR_BGR2GRAY versions.
     */
    [[nodiscard]] auto load_images() -> bool;

    // --------------------------------------------------------------
    // BASIC COMPARISON
    // --------------------------------------------------------------

    [[nodiscard]] auto compare_basic() const -> std::string;

    // --------------------------------------------------------------
    // COLOR-SPACE ANALYSIS
    // --------------------------------------------------------------

    [[nodiscard]] auto compare_color_space() const -> std::string;

    // --------------------------------------------------------------
    // HISTOGRAMS
    // --------------------------------------------------------------

    [[nodiscard]] auto compute_histogram() const -> HistogramResult;
    [[nodiscard]] auto summarize(const HistogramResult& r) const -> std::string;
    [[nodiscard]] auto compare_histogram() const -> std::string;

    // --------------------------------------------------------------
    // STRUCTURAL SIMILARITY
    // --------------------------------------------------------------

    [[nodiscard]] auto compute_structural() const -> StructuralResult;
    [[nodiscard]] auto summarize(const StructuralResult& r) const -> std::string;
    [[nodiscard]] auto compare_structural() const -> std::string;

    // --------------------------------------------------------------
    // FEATURES (ORB / AKAZE / SIFT)
    // --------------------------------------------------------------

    /**
     * @brief Extracts keypoints, descriptors, matches and geometric consistency.
     *
     * Returns a complete FeatureResult for GUI consumption.
     */
    [[nodiscard]] auto compute_features(FeatureMethod method = FeatureMethod::AKAZE) const -> FeatureResult;

    /**
     * @brief Human-readable summary of FeatureResult.
     */
    [[nodiscard]] auto summarize(const FeatureResult& r) const -> std::string;

    /**
     * @brief Full feature-based comparison pipeline.
     *
     * Performs:
     *   - Detection
     *   - Description
     *   - KNN matching
     *   - Lowe ratio filtering
     *   - RANSAC homography estimation
     *   - Similarity metrics
     */
    [[nodiscard]] auto compare_features(FeatureMethod method = FeatureMethod::AKAZE) const -> std::string;

    // --------------------------------------------------------------
    // REPORT EXPORT
    // --------------------------------------------------------------

    auto export_report(const std::filesystem::path& output_path,
                       FeatureMethod method = FeatureMethod::AKAZE) const -> bool;

    auto export_report(std::string_view output_path,
                       FeatureMethod method = FeatureMethod::AKAZE) -> bool;

    // --------------------------------------------------------------
    // ACCESSORS
    // --------------------------------------------------------------

    auto images() const -> std::pair<cv::Mat, cv::Mat>;
    auto paths() const -> std::pair<std::filesystem::path, std::filesystem::path>;

private:
    cv::Mat image1_;
    cv::Mat image2_;
    cv::Mat grayscale1_;
    cv::Mat grayscale2_;

    std::filesystem::path path1_;
    std::filesystem::path path2_;
};

// --------------------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------------------

void save(std::string_view filepath, const cv::Mat& image);
auto load(std::string_view filepath) -> cv::Mat;
auto exists(std::string_view filepath) -> bool;

} // namespace imgtools

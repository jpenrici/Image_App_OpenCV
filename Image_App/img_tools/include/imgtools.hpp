#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include <expected>
#include <filesystem>
#include <utility>

namespace imgtools {

/**
 * @class ImageAnalyzer
 * @brief Utility class for performing multiple image comparison techniques.
 *
 * This class provides a unified interface for loading two images from disk
 * and performing several types of comparisons:
 *
 * - Basic metadata comparison (dimensions, channels, bit depth).
 * - Color space analysis (RGB, HSV, LAB).
 * - Histogram-based similarity (correlation, chi-square, intersection, Bhattacharyya, KL divergence).
 * - Structural similarity (MSE, PSNR, SSIM).
 * - Feature-based comparison (ORB keypoints, matching quality — optional).
 *
 * All comparison methods return descriptive strings suitable for console output,
 * logging, or exporting to a report file.
 */
class ImageAnalyzer {
public:
    /**
     * @brief Constructs an ImageAnalyzer with two image paths.
     *
     * The images are not loaded automatically. Call load_images()
     * prior to running any comparison functions.
     *
     * @param path1 Path to the first image.
     * @param path2 Path to the second image.
     */
    explicit ImageAnalyzer(std::string_view path1, std::string_view path2) noexcept;
    ~ImageAnalyzer() noexcept = default;

    ImageAnalyzer(const ImageAnalyzer&) = default;
    auto operator=(const ImageAnalyzer&) -> ImageAnalyzer& = default;

    ImageAnalyzer(ImageAnalyzer&&) noexcept = default;
    auto operator=(ImageAnalyzer&&) noexcept -> ImageAnalyzer& = default;

    /**
     * @brief Loads both images from disk.
     *
     * On success:
     *  - image1_ and image2_ hold the full-color images.
     *  - grayscale1_ and grayscale2_ store their grayscale versions.
     *
     * @return true if both files exist and were loaded correctly,
     *         false otherwise.
     */
    [[nodiscard]] auto load_images() noexcept -> bool;

    /**
     * @brief Performs basic comparison between images.
     *
     * This includes:
     *  - Resolution
     *  - Number of channels
     *  - Bit depth
     *  - File paths and existence checks
     *
     * @return Human-readable formatted string with metadata comparison.
     */
    [[nodiscard]] auto compare_basic() const -> std::string;

    /**
     * @brief Compares images across multiple color spaces.
     *
     * Converts both images into:
     *  - RGB (if not already)
     *  - HSV
     *  - LAB
     *
     * For each color space:
     *  - Mean and standard deviation of each channel are computed.
     *
     * @return Formatted report showing differences in color distribution.
     */
    [[nodiscard]] auto compare_color_space() const -> std::string;

    /**
     * @brief Compares grayscale histograms between the two images.
     *
     * Computes:
     *  - Correlation                (1.0 = identical, -1.0 = inverse)
     *  - Chi-Square                 (0 = identical)
     *  - Intersection               (higher = more similar)
     *  - Bhattacharyya distance     (0 = identical)
     *  - Kullback–Leibler divergence
     *
     * The report includes automatically interpreted quality ranges.
     *
     * @return Detailed histogram similarity analysis.
     */
    [[nodiscard]] auto compare_histogram() const -> std::string;

    /**
     * @brief Structural similarity comparison.
     *
     * Structural comparison goes beyond raw pixel differences and attempts
     * to evaluate how similar two images are in perceptual terms.
     *
     * The following measures are calculated:
     *
     *  - **MSE (Mean Squared Error)**:
     *       Measures raw pixel error (0 = perfect image).
     *
     *  - **PSNR (Peak Signal-to-Noise Ratio)**:
     *       Measures perceptual quality in decibels
     *       (>40 dB excellent, 20–30 dB moderate, <20 dB poor).
     *
     *  - **SSIM (Structural Similarity Index)**:
     *       Measures contrast, luminance, and structure similarity.
     *       (1.0 = perfect, 0 = unrelated, <0 = negative correlation/inversion).
     *
     * The report includes:
     *  - Interpretation text (excellent, good, degraded, etc.)
     *  - Automatic detection of inverse-structure patterns.
     *
     * @return Formatted structural comparison report.
     */
    [[nodiscard]] auto compare_structural() const -> std::string;

    /**
     * @brief Methods available for local feature detection and description.
     *
     * ORB  - Fast, efficient, binary descriptors. Works well for real-time tasks.
     * AKAZE - Nonlinear scale space; robust, stable, excellent for general matching.
     * SIFT - High-accuracy float descriptors; best for reliability, slower.
     */
    enum class FeatureMethod { ORB, AKAZE, SIFT };

    /**
     * @brief Feature-based comparison using local keypoint descriptors.
     *
     * This function extracts local features from both images using the selected
     * detection/description method and evaluates their structural similarity.
     *
     * Supported algorithms:
     *  - ORB   (binary, fast, NORM_HAMMING)
     *  - AKAZE (binary, robust, NORM_HAMMING)
     *  - SIFT  (float descriptors, high accuracy, NORM_L2)
     *
     * Processing pipeline:
     *  1. Detect keypoints in each image.
     *  2. Compute local descriptors.
     *  3. Perform KNN descriptor matching (k = 2).
     *  4. Apply the Lowe ratio test to filter ambiguous matches.
     *  5. Estimate geometric consistency using RANSAC homography.
     *  6. Classify transformation type (affine / perspective).
     *  7. Compute match confidence metrics:
     *        - Inlier/Outlier ratio
     *        - Average match distance
     *        - Variance of distances
     *        - Mean Lowe ratio
     *  8. Generate a human-readable textual summary of similarity.
     *
     * Typical use cases:
     *  - Detecting geometric transformations (rotation, scaling, perspective)
     *  - Comparing structural similarity between objects
     *  - Matching images with different illumination or partial occlusion
     *  - Validating whether two images represent the same scene
     *
     * Notes:
     *  - Images are internally compared in grayscale.
     *  - SIFT requires OpenCV built with the xfeatures2d contrib module.
     *  - A minimum of 4 matches after filtering is required for homography.
     *
     * @param method Feature extraction method (ORB, AKAZE, or SIFT).
     * @return String containing a detailed feature-matching report.
     */
    [[nodiscard]] auto compare_features(FeatureMethod method = FeatureMethod::AKAZE) const -> std::string;

    /**
     * @brief Exports the combined comparison report into a text file.
     *
     * The exported report contains:
     *  - Basic comparison
     *  - Color-space differences
     *  - Histogram analysis
     *  - Structural comparison
     *  - Feature-based evaluation
     *
     * @param output_path Path to write the report to.
     * @return true if successful, false if file could not be written.
     */
    auto export_report(const std::filesystem::path& output_path,
                       FeatureMethod method = FeatureMethod::AKAZE) const -> bool;

    /**
     * @brief Convenience overload for export_report().
     */
    auto export_report(std::string_view output_path,
                       FeatureMethod method = FeatureMethod::AKAZE) -> bool;

    /**
     * @brief Returns the loaded full-color images.
     *
     * @return A pair containing (image1_, image2_).
     */
    auto images() const -> std::pair<cv::Mat, cv::Mat>;

    /**
     * @brief Returns the paths passed to the constructor.
     *
     * @return (path1_, path2_).
     */
    auto paths() const -> std::pair<std::filesystem::path, std::filesystem::path>;

private:
    cv::Mat image1_;       //!< Original first image.
    cv::Mat image2_;       //!< Original second image.
    cv::Mat grayscale1_;   //!< Grayscale version of first image.
    cv::Mat grayscale2_;   //!< Grayscale version of second image.
    std::filesystem::path path1_; //!< Path to first image.
    std::filesystem::path path2_; //!< Path to second image.
};

// --- Helper functions ---

/**
 * @brief Saves an image to disk.
 *
 * @param filepath Destination file path.
 * @param image The image to save.
 */
void save(std::string_view filepath, const cv::Mat& image);

/**
 * @brief Loads an image from disk.
 *
 * On failure returns an empty cv::Mat.
 *
 * @param filepath Path of the image to load.
 * @return The loaded image or empty matrix on error.
 */
auto load(std::string_view filepath) -> cv::Mat;

/**
 * @brief Checks whether a file exists.
 *
 * @param filepath Path string.
 * @return true if file exists.
 */
auto exists(std::string_view filepath) -> bool;

} // namespace imgtools

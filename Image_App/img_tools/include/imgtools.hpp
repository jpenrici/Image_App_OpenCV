#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include <expected>
#include <filesystem>

namespace imgtools {

class ImageAnalyzer {
public:
    explicit ImageAnalyzer(std::string_view path1, std::string_view path2) noexcept;
    ~ImageAnalyzer() noexcept = default;

    ImageAnalyzer(const ImageAnalyzer&) = delete;
    auto operator=(const ImageAnalyzer&) -> ImageAnalyzer& = delete;

    ImageAnalyzer(ImageAnalyzer&&) noexcept = default;
    auto operator=(ImageAnalyzer&&) noexcept -> ImageAnalyzer& = default;

    [[nodiscard]] auto load_images() noexcept -> bool;
    [[nodiscard]] auto compare_basic() const -> std::string;
    [[nodiscard]] auto compare_color_space() const -> std::string;
    [[nodiscard]] auto compare_histogram() const -> std::string;
    [[nodiscard]] auto compare_structural() const -> std::string;
    [[nodiscard]] auto compare_features() const -> std::string;

    auto export_report(const std::filesystem::path& output_path) const -> bool;

private:
    cv::Mat image1_;
    cv::Mat image2_;
    std::filesystem::path path1_;
    std::filesystem::path path2_;
};

// --- Helper functions ---

// Save an OpenCV image to disk.
void save(std::string_view filepath, const cv::Mat& image);

// Load an OpenCV image from disk.
auto load(std::string_view filepath) -> cv::Mat;

// Check if a file exists.
auto exists(std::string_view filepath) -> bool;

}

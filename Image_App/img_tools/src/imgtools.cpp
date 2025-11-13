#include "imgtools.hpp"

#include <chrono>
#include <fstream>
#include <print>
#include <sstream>

namespace imgtools {

// --- Helper functions implementation ---

void save(std::string_view filepath, const cv::Mat &image) {
  // Save image to disk using OpenCV.
  cv::imwrite(std::string(filepath), image);
}

auto load(std::string_view filepath) -> cv::Mat {
  // Load image using OpenCV.
  return cv::imread(std::string(filepath), cv::IMREAD_UNCHANGED);
}

auto exists(std::string_view filepath) -> bool {
  // Check if file exists in filesystem.
  return std::filesystem::exists(filepath);
}

// --- Class implementation ---

ImageAnalyzer::ImageAnalyzer(std::string_view path1,
                             std::string_view path2) noexcept
    : path1_(path1), path2_(path2) {}

// Load both images safely.
auto ImageAnalyzer::load_images() noexcept -> bool {
  image1_ = cv::imread(path1_.string(), cv::IMREAD_UNCHANGED);
  image2_ = cv::imread(path2_.string(), cv::IMREAD_UNCHANGED);

  return !image1_.empty() && !image2_.empty();
}

// Compare file extension, file size, dimensions and channels.
auto ImageAnalyzer::compare_basic() const -> std::string {
  std::ostringstream oss;

  oss << "[Basic Comparison]\n";

  // Compare extensions.
  auto ext1 = path1_.extension().string();
  auto ext2 = path2_.extension().string();
  oss << "Extension 1: " << ext1 << "\n";
  oss << "Extension 2: " << ext2 << "\n";
  oss << "Extension match: " << std::boolalpha << (ext1 == ext2) << "\n";

  // Compare file sizes.
  auto size1 = std::filesystem::file_size(path1_);
  auto size2 = std::filesystem::file_size(path2_);
  oss << "File size 1: " << size1 << " bytes\n";
  oss << "File size 2: " << size2 << " bytes\n";
  oss << "Size difference: "
      << static_cast<long long>(size1) - static_cast<long long>(size2)
      << " bytes\n";

  // Compare image metadata.
  if (!image1_.empty() && !image2_.empty()) {
    oss << "Dimensions 1: " << image1_.cols << "x" << image1_.rows << "\n";
    oss << "Dimensions 2: " << image2_.cols << "x" << image2_.rows << "\n";
    oss << "Channels 1: " << image1_.channels() << "\n";
    oss << "Channels 2: " << image2_.channels() << "\n";
  } else {
    oss << "Images not loaded properly.\n";
  }

  return oss.str();
}

// Simple color space analysis.
auto ImageAnalyzer::compare_color_space() const -> std::string {
  std::ostringstream oss;

  oss << "[Color Space]\n";
  if (image1_.empty() || image2_.empty()) {
    oss << "Images not loaded.\n";
    return oss.str();
  }

  auto describe = [](const cv::Mat &img) -> std::string {
    switch (img.channels()) {
    case 1:
      return "Grayscale";
    case 3:
      return "RGB";
    case 4:
      return "RGBA";
    default:
      return "Unknown";
    }
  };

  auto type1 = describe(image1_);
  auto type2 = describe(image2_);

  oss << "Image 1: " << type1 << "\n";
  oss << "Image 2: " << type2 << "\n";
  oss << "Color space match: " << std::boolalpha << (type1 == type2) << "\n";

  return oss.str();
}

// Export a detailed report with headers and separators.
auto ImageAnalyzer::export_report(
    const std::filesystem::path &output_path) const -> bool {

  std::ofstream file(output_path);
  if (!file.is_open())
    return false;

  std::ostringstream oss;

  // Header
  oss << "========================================\n";
  oss << "[Image Comparison Report]\n";
  oss << "========================================\n\n";

  oss << "[Images]\n";
  oss << "Path 1: " << path1_ << "\n";
  oss << "Path 2: " << path2_ << "\n";

  // Timestamp (opcional)
  auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  oss << "Date:   " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S")
      << "\n\n";

  // Append comparisons
  oss << compare_basic() << "\n";
  oss << compare_color_space() << "\n";

  // Footer
  oss << "----------------------------------------\n";
  oss << "End of report.\n";

  file << oss.str();
  file.close();

  return true;
}

} // namespace imgtools

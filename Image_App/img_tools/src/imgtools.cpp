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

auto ImageAnalyzer::paths() const
    -> std::pair<std::filesystem::path, std::filesystem::path> {
  return {path1_, path2_};
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

auto ImageAnalyzer::compare_histogram() const -> std::string {

  if (image1_.empty() || image2_.empty()) {
    return "[Histogram]\nFailed: One or both images are not loaded.\n";
  }

  // --- Step 1: Convert BGR images to HSV ---
  // OpenCV uses BGR (Blue Green Red) instead of the traditional RGB sequence.
  // HSV makes histogram comparison more robust to brightness differences.
  cv::Mat hsv1, hsv2;
  cv::cvtColor(image1_, hsv1, cv::COLOR_BGR2HSV);
  cv::cvtColor(image2_, hsv2, cv::COLOR_BGR2HSV);

  // --- Step 2: Histogram parameters ---
  // Number of bins for Hue and Saturation. More bins = more precision.
  int h_bins{50};
  int s_bins{60};
  int hist_size[] = {h_bins, s_bins};

  // Hue ranges from 0 to 179 in OpenCV.
  // Saturation ranges from 0 to 255.
  float h_range[] = {0.f, 180.f};
  float s_range[] = {0.f, 256.f};
  const float *ranges[] = {h_range, s_range};

  // It compares only the H and S channels (channels 0 and 1 in HSV).
  int channels[] = {0, 1};

  cv::Mat hist1, hist2;

  // --- Step 3: Compute 2D histograms ---
  // cv::calcHist creates a histogram using the selected channels and ranges.
  cv::calcHist(&hsv1, 1, channels, cv::Mat(), hist1, 2, hist_size, ranges, true,
               false);
  cv::calcHist(&hsv2, 1, channels, cv::Mat(), hist2, 2, hist_size, ranges, true,
               false);

  // --- Step 4: Normalize histograms ---
  // Normalization ensures the comparison does not depend on image size.
  cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX);
  cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX);

  // --- Step 5: Compute similarity metrics ---
  // Multiple metrics give more insight about the histogram similarity.
  double corr = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
  double chisq = cv::compareHist(hist1, hist2, cv::HISTCMP_CHISQR);
  double inter = cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
  double bhatt = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);

  // --- Step 6: Build output string ---
  std::ostringstream oss;
  oss << "[Histogram]\n";
  oss << "Correlation:       " << corr << "\n";
  oss << "Chi-Square:        " << chisq << "\n";
  oss << "Intersection:      " << inter << "\n";
  oss << "Bhattacharyya:     " << bhatt << "\n";

  // --- Step 7: Basic interpretation based on correlation value ---
  if (corr > 0.90)
    oss << "Conclusion: Very similar histograms.\n";
  else if (corr > 0.70)
    oss << "Conclusion: Moderately similar histograms.\n";
  else
    oss << "Conclusion: Different histograms.\n";

  return oss.str();
}

// Export a detailed report with headers and separators.
auto ImageAnalyzer::export_report(
    const std::filesystem::path &output_path) const -> bool {

  std::string path{output_path};
  if (std::filesystem::path(output_path).extension() != ".txt") {
    path.append(".txt");
  }

  std::ofstream file(path);
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

  // Timestamp
  auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  oss << "Date:   " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S")
      << "\n\n";

  // Append comparisons
  oss << compare_basic() << "\n";
  oss << compare_color_space() << "\n";
  oss << compare_histogram() << "\n";

  // Footer
  oss << "----------------------------------------\n";
  oss << "End of report.\n";

  file << oss.str();
  file.close();

  return true;
}

auto ImageAnalyzer::export_report(std::string_view output_path) -> bool {
  return export_report(std::filesystem::path(output_path));
}

} // namespace imgtools

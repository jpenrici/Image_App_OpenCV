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

  // --- Step 1: Validate images ---
  if (image1_.empty() || image2_.empty()) {
    return "[Histogram]\nFailed: One or both images are not loaded.\n";
  }

  // --- Step 2: Convert safely to grayscale ---
  // This guarantees full compatibility with:
  // - 1 - channel grayscale
  // - 3 - channel BGR
  // - 4 - channel BGRA
  cv::Mat gray1, gray2;

  auto to_gray = [](const cv::Mat &src, cv::Mat &dst) {
    if (src.channels() == 1) {
      dst = src.clone(); // already grayscale
    } else if (src.channels() == 3) {
      cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
      cv::cvtColor(src, dst, cv::COLOR_BGRA2GRAY);
    } else {
      throw std::runtime_error("Unsupported number of channels for histogram.");
    }
  };

  try {
    to_gray(image1_, gray1);
    to_gray(image2_, gray2);
  } catch (const std::exception &ex) {
    return std::string("[Histogram]\nError: ") + ex.what() + "\n";
  }

  // --- Step 3: Histogram configuration (1D grayscale) ---
  int histSize = 256;       // bins
  float range[] = {0, 256}; // pixel values
  const float *histRange = range;
  int channel = 0; // grayscale = single channel

  cv::Mat hist1, hist2;

  // --- Step 4: Compute histograms ---
  // cv::calcHist creates a histogram using the selected channels and ranges.
  cv::calcHist(&gray1, 1, &channel, cv::Mat(), hist1, 1, &histSize, &histRange,
               true, false);

  cv::calcHist(&gray2, 1, &channel, cv::Mat(), hist2, 1, &histSize, &histRange,
               true, false);

  // --- Step 5: Normalize for comparability ---
  // Normalization ensures the comparison does not depend on image size.
  cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX);
  cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX);

  // --- Step 6: Compute similarity metrics ---
  // Multiple metrics give more insight about the histogram similarity.
  double corr = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
  double chisq = cv::compareHist(hist1, hist2, cv::HISTCMP_CHISQR);
  double inter = cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
  double bhatt = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);

  // --- Step 7: Build output string ---
  std::ostringstream oss;
  oss << "[Histogram]\n";
  oss << "Correlation:       " << corr << "\n";
  oss << "Chi-Square:        " << chisq << "\n";
  oss << "Intersection:      " << inter << "\n";
  oss << "Bhattacharyya:     " << bhatt << "\n";

  // --- Step 8: Basic interpretation based on correlation value ---
  oss << "Histogram Correlation: " << std::format("{:.4f}", corr) << "\n";

  oss << "Similarity Score: " << std::format("{:.2f}%", corr * 100.0) << "\n";

  oss << "Interpretation: ";
  if (corr > 0.85)
    oss << "Histograms are nearly identical";
  else if (corr > 0.50)
    oss << "Strong similarity between histograms";
  else if (corr > 0.20)
    oss << "Moderate similarity";
  else if (corr > -0.20)
    oss << "No correlation";
  else if (corr > -0.50)
    oss << "Moderate inverse relation";
  else
    oss << "Histograms are inverses of each other";

  oss << "\n";

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

#include "imgtools.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <string_view>

const std::string TEST_PATH = "resource/test";

inline std::string fpath(const std::string_view filename) {
  return (std::filesystem::path(TEST_PATH) / filename).string();
}

void image() {
  const std::filesystem::path dir{TEST_PATH};
  if (std::filesystem::create_directories(dir)) {
    std::println("Directory created for testing!");
  }

  const int width = 256;
  const int height = 256;

  std::array<cv::Mat, 6> img;
  img.at(0) = cv::Mat();                       // empty
  img.at(1) = cv::Mat(height, width, CV_8UC1); // Grayscale
  img.at(3) = cv::Mat(height, width, CV_8UC1);

  // Build gradient: left = 0, right = 255
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img.at(1).at<uchar>(y, x) = static_cast<uchar>(x); // 0..255
      img.at(3).at<uchar>(y, x) =
          255 - static_cast<uchar>(x); // inverted gradient
    }
  }

  // Identical images
  img.at(2) = img.at(1);

  // Colorful images
  img.at(4) = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 255)); // Red
  img.at(5) = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue

  // Save
  for (size_t i = 1; i <= 5; i++) {
    auto p = std::format("{}/img{}.png", TEST_PATH, i);
    imgtools::save(p, img.at(i));
  }
}

auto test_basic(imgtools::ImageAnalyzer &img) -> std::string {

  std::ostringstream oss;
  oss << "\n---- Basic Metadata ----\n";
  oss << img.compare_basic();
  oss << "\n---- Color Space ----\n";
  oss << img.compare_color_space();

  return oss.str();
}

auto test_histogram(imgtools::ImageAnalyzer &img) -> std::string {
  std::ostringstream oss;
  oss << "---- Histogram Analysis ----\n";
  oss << img.compare_histogram();

  return oss.str();
}

auto test_structural(imgtools::ImageAnalyzer &img) -> std::string {
  std::ostringstream oss;
  oss << "---- Structural Analysis ----\n";
  oss << img.compare_structural();

  return oss.str();
}

void analyze(std::pair<std::string_view, std::string_view> paths) {

  auto [path1, path2] = paths;
  imgtools::ImageAnalyzer iia(path1, path2);

  if (!imgtools::exists(path1) || !imgtools::exists(path2)) {
    std::println("Images missing. Aborting test!");
    return;
  }

  if (iia.load_images()) {
    auto name1 = std::filesystem::path(path1).filename().string();
    auto name2 = std::filesystem::path(path2).filename().string();

    std::println("========================================");
    std::println("Test: Comparing '{}' <-> '{}'", name1, name2);
    std::println("========================================\n");

    std::println("{}", test_basic(iia));
    std::println("{}", test_histogram(iia));
    std::println("{}", test_structural(iia));

    auto outname = std::format("report_{}_{}", name1.substr(0, name1.find('.')),
                               name2.substr(0, name2.find('.')));

    iia.export_report(fpath(outname));
  }
}

void test() {
  std::println("Start test ...");

  // Create images for testing
  image();

  // Analyze
  auto check = [](std::string_view p1, std::string_view p2) {
    analyze({fpath(p1), fpath(p2)});
  };

  // Test - identical images
  check("img1.png", "img2.png");

  // Test - different images
  check("img1.png", "img3.png");
  check("img1.png", "img4.png");
  check("img4.png", "img5.png");

  std::println("Test completed.");
}

auto main() -> int {

  test();

  return EXIT_SUCCESS;
}

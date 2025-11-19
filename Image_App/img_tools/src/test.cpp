#include "imagetools.hpp"

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

using Result = std::expected<std::string, std::string>;

const std::string MSG_ERROR = "Images not available.";

void image() {
  const std::filesystem::path dir{TEST_PATH};
  if (std::filesystem::create_directories(dir)) {
    std::println("Directory created for testing!");
  }

  const int width = 256;
  const int height = 256;

  std::array<cv::Mat, 8> img;
  img.at(0) = cv::Mat();                       // empty
  img.at(1) = cv::Mat(height, width, CV_8UC1); // Grayscale
  img.at(2) = img.at(1);                       // Identical images
  img.at(3) = cv::Mat(height, width, CV_8UC1); // Grayscale

  // Build gradient: left = 0, right = 255
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      img.at(1).at<uchar>(y, x) = static_cast<uchar>(x); // 0..255
      img.at(3).at<uchar>(y, x) =
          255 - static_cast<uchar>(x); // inverted gradient
    }
  }

  // Colorful images
  img.at(4) = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 255)); // Red
  img.at(5) = cv::Mat(height, width, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue

  // Geometric shapes
  img.at(6) =
      cv::Mat(height, width, CV_8UC1, cv::Scalar(0)); // black background
  img.at(7) =
      cv::Mat(height, width, CV_8UC1, cv::Scalar(0)); // black background

  // Draw a centered white square
  cv::rectangle(img.at(6), cv::Point(80, 80), cv::Point(176, 176),
                cv::Scalar(255), cv::FILLED);

  // Draw a shifted version of the square
  cv::rectangle(img.at(7), cv::Point(100, 90), cv::Point(196, 186),
                cv::Scalar(255), cv::FILLED);

  // Small rotation for image to create a better feature test
  double angle = 45.0; // degree rotation
  cv::Point2f center(width / 2.0f, height / 2.0f);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  cv::warpAffine(img.at(7), img.at(7), rot, img.at(7).size());

  // Save
  for (size_t i = 1; i <= 7; i++) {
    auto p = std::format("{}/img{}.png", TEST_PATH, i);
    imgtools::save(p, img.at(i));
  }
}

auto test_basic(imgtools::ImageAnalyzer &img) -> Result {

  std::ostringstream oss;
  oss << "\n---- Test : Basic Metadata and Color Space ----\n";

  if (!img.images_available()) {
    oss << MSG_ERROR << "\n";
    return std::unexpected(oss.str());
  }

  oss << img.compare_basic() << "\n";
  oss << img.compare_color_space();

  return oss.str();
}

auto test_histogram(imgtools::ImageAnalyzer &img) -> Result {

  std::ostringstream oss;
  oss << "---- Test : Histogram Analysis ----\n";

  if (!img.images_available()) {
    oss << MSG_ERROR << "\n";
    return std::unexpected(oss.str());
  }

  oss << img.compare_histogram();

  return oss.str();
}

auto test_structural(imgtools::ImageAnalyzer &img) -> Result {

  std::ostringstream oss;
  oss << "---- Test : Structural Analysis ----\n";

  if (!img.images_available()) {
    oss << MSG_ERROR << "\n";
    return std::unexpected(oss.str());
  }

  oss << img.compare_structural();

  return oss.str();
}

auto test_features(imgtools::ImageAnalyzer &img, imgtools::FeatureMethod method)
    -> Result {

  std::ostringstream oss;
  oss << "---- Test : Features Analysis ----\n";

  if (!img.images_available()) {
    oss << MSG_ERROR << "\n";
    return std::unexpected(oss.str());
  }

  oss << img.compare_features(method);

  return oss.str();
}

void analyze(std::pair<std::string_view, std::string_view> paths,
             imgtools::FeatureMethod method = imgtools::FeatureMethod::AKAZE) {

  auto [path1, path2] = paths;
  imgtools::ImageAnalyzer iia(path1, path2);

  auto name1 = std::filesystem::path(path1).filename().string();
  auto name2 = std::filesystem::path(path2).filename().string();

  std::println("========================================");
  std::println("Test: Comparing '{}' <-> '{}'", name1, name2);
  std::println("========================================");

  auto process = [](Result r) {
    if (r.has_value()) {
      std::println("{}", r.value());
    } else {
      std::println("{}", r.error());
    }
  };

  process(test_basic(iia));
  process(test_histogram(iia));
  process(test_structural(iia));
  process(test_features(iia, method));

  auto outname = std::format("report_{}_{}", name1.substr(0, name1.find('.')),
                             name2.substr(0, name2.find('.')));

  iia.export_report(fpath(outname), method);

  std::println("----------------------------------------\n");
}

void test() {
  std::println("Start test ...");

  // Create images for testing
  image();

  // Analyze
  auto check = [](std::string_view p1, std::string_view p2) {
    analyze({fpath(p1), fpath(p2)});
  };

  // Test - invalid images
  check("", "");

  // Test - identical images
  check("img1.png", "img2.png");

  // Test - different images
  check("img1.png", "img3.png");
  check("img1.png", "img4.png");
  check("img4.png", "img5.png");
  check("img5.png", "img6.png");

  // Test - similar images
  check("img6.png", "img7.png");

  analyze({fpath("img7.png"), fpath("img6.png")}, imgtools::FeatureMethod::ORB);

  std::println("Test completed.");
}

auto main() -> int {

  test();

  return EXIT_SUCCESS;
}

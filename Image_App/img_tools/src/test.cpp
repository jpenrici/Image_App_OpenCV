#include "imgtools.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <string_view>
#include <tuple>

inline std::string f(const std::string_view dir,
                     const std::string_view filename) {
  return (std::filesystem::path(dir) / filename).string();
}

const std::string TEST_PATH = "resource/test";
const auto IMG1_PATH = f(TEST_PATH, "imgA.png");
const auto IMG2_PATH = f(TEST_PATH, "imgB.png");
const auto IMG3_PATH = f(TEST_PATH, "imgC.png");

void image() {
  const std::filesystem::path dir{TEST_PATH};
  if (std::filesystem::create_directories(dir)) {
    std::println("Directory created for testing!");
  }

  cv::Mat imgA(200, 200, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
  cv::Mat imgB(200, 200, CV_8UC3, cv::Scalar(0, 0, 255)); // Red

  imgtools::save(IMG1_PATH, imgA);
  imgtools::save(IMG2_PATH, imgA);
  imgtools::save(IMG3_PATH, imgB);
}

void test_basic(
    std::tuple<std::string_view, std::string_view, std::string_view> paths) {

  auto [path1, path2, output_name] = paths;

  std::println("Testing basic comparison between '{}' and '{}'\n",
               std::filesystem::path(path1).filename().string(),
               std::filesystem::path(path2).filename().string());

  imgtools::ImageAnalyzer analyzer(path1, path2);
  if (!analyzer.load_images()) {
    std::println("Failed to load images!");
    return;
  }

  std::println("{}", analyzer.compare_basic());
  std::println("{}", analyzer.compare_color_space());

  std::string filename{output_name};
  if (std::filesystem::path(output_name).extension() != ".txt") {
    filename.append(".txt");
  }
  analyzer.export_report(f(TEST_PATH, filename));
}

void test() {
  std::println("Start test ...");

  // Create images for testing
  image();

  // Check paths
  if (!imgtools::exists(IMG1_PATH) || !imgtools::exists(IMG2_PATH) ||
      !imgtools::exists(IMG3_PATH)) {
    std::println("Images missing. Aborting test!");
    return;
  }

  // Basic Test
  test_basic({IMG1_PATH, IMG2_PATH, "result_img1_img2"}); // identical images
  test_basic({IMG1_PATH, IMG3_PATH, "result_img1_img3"}); // different images

  std::println("Test completed.");
}

auto main() -> int {

  test();

  return EXIT_SUCCESS;
}

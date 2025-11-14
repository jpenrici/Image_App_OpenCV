#include "imgtools.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <string_view>

inline std::string f(const std::string_view dir,
                     const std::string_view filename) {
  return (std::filesystem::path(dir) / filename).string();
}

const std::string TEST_PATH = "resource/test";
const std::string IMG1_PATH = f(TEST_PATH, "imgA.png");
const std::string IMG2_PATH = f(TEST_PATH, "imgB.png");
const std::string IMG3_PATH = f(TEST_PATH, "imgC.png");

void image() {
  const std::filesystem::path dir{TEST_PATH};
  if (std::filesystem::create_directories(dir)) {
    std::println("Directory created for testing!");
  }

  cv::Mat img1(200, 200, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
  cv::Mat img2 = img1;
  cv::Mat img3(200, 200, CV_8UC3, cv::Scalar(0, 0, 255)); // Red

  imgtools::save(IMG1_PATH, img1);
  imgtools::save(IMG2_PATH, img2);
  imgtools::save(IMG3_PATH, img3);
}

auto test_basic(imgtools::ImageAnalyzer &img) -> std::string {

  std::string result;
  result.append(img.compare_basic());
  result.append(img.compare_color_space());

  return result;
}

auto test_histogram(imgtools::ImageAnalyzer &img) -> std::string {
  return img.compare_histogram();
}

void analyze(std::pair<std::string_view, std::string_view> paths,
             std::string_view output_name) {

  auto [path1, path2] = paths;
  imgtools::ImageAnalyzer a(path1, path2);

  if (a.load_images()) {
    std::println("Testing basic comparison between '{}' and '{}'\n",
                 std::filesystem::path(path1).filename().string(),
                 std::filesystem::path(path2).filename().string());

    std::println("{}", test_basic(a));
    std::println("{}", test_histogram(a));

    a.export_report(f(TEST_PATH, output_name));
  }
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

  // Test - identical images
  analyze({IMG1_PATH, IMG2_PATH}, "result_img1_img2");

  // Test - different images
  analyze({IMG1_PATH, IMG3_PATH}, "result_img1_img3");

  std::println("Test completed.");
}

auto main() -> int {

  test();

  return EXIT_SUCCESS;
}

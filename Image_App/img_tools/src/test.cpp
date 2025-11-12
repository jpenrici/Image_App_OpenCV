#include "imgtools.hpp"

#include <cstdlib>
#include <expected>
#include <filesystem>
#include <print>
#include <string_view>

using namespace imgtools;

void image() {
  // Create three test images (two identical, one different).
  const std::filesystem::path dir = "build/test";
  std::filesystem::create_directories(dir);

  cv::Mat imgA(200, 200, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
  cv::Mat imgB = imgA.clone();                            // Identical
  cv::Mat imgC(200, 200, CV_8UC3, cv::Scalar(0, 0, 255)); // Red

  save(dir / "imgA.png", imgA);
  save(dir / "imgB.png", imgB);
  save(dir / "imgC.png", imgC);
}

void test_basic(std::string_view path1, std::string_view path2) {
  std::println("Testing basic comparison between '{}' and '{}'", path1, path2);

  ImageAnalyzer analyzer(path1, path2);
  if (!analyzer.load_images()) {
    std::println("Failed to load images.");
    return;
  }

  std::println("{}", analyzer.compare_basic());
  std::println("{}", analyzer.compare_color_space());
  analyzer.export_report("build/test/report_basic.txt");
}

void test() {
  std::println("Start test ...");

  image();

  const std::string imagePath1 = "build/test/imgA.png";
  const std::string imagePath2 = "build/test/imgB.png";
  const std::string imagePath3 = "build/test/imgC.png";

  if (!exists(imagePath1) || !exists(imagePath2) || !exists(imagePath3)) {
    std::println("Images missing. Aborting test.");
    return;
  }

  test_basic(imagePath1, imagePath2); // identical images
  test_basic(imagePath1, imagePath3); // different images

  std::println("Test completed.");
}

auto main() -> int {
  test();
  return EXIT_SUCCESS;
}

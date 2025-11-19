#pragma once

#include "imagetools.hpp"

/**
 * @brief The ImageAnalyzerGui class
 * ImageAnalyzer Wrapper
 */
class ImageAnalyzerGui {

public:
  ImageAnalyzerGui(std::string_view path1, std::string_view path2);

  auto available() const noexcept -> bool;

  auto analyzer() noexcept -> imgtools::ImageAnalyzer *;

  auto path1() const noexcept -> const std::filesystem::path &;

  auto path2() const noexcept -> const std::filesystem::path &;

private:
  std::filesystem::path current_path1_;
  std::filesystem::path current_path2_;

  bool status_{false};

  std::unique_ptr<imgtools::ImageAnalyzer> analyzer_;
};

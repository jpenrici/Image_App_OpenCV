#include "imagetoolsgui.hpp"

ImageAnalyzerGui::ImageAnalyzerGui(std::string_view path1,
                                   std::string_view path2) {}

auto ImageAnalyzerGui::available() const noexcept -> bool {
  return false; // TO DO
}

auto ImageAnalyzerGui::analyzer() noexcept -> imgtools::ImageAnalyzer * {
  return nullptr; // TO DO
}

auto ImageAnalyzerGui::path1() const noexcept -> const std::filesystem::path {
  const std::filesystem::path a("to do");
  return a;
}
auto ImageAnalyzerGui::path2() const noexcept -> const std::filesystem::path {
  const std::filesystem::path a("to do");
  return a;
}

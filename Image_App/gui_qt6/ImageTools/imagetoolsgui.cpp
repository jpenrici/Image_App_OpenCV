#include "imagetoolsgui.hpp"

ImageAnalyzerGui::ImageAnalyzerGui(std::string_view path1,
                                   std::string_view path2)
    : current_path1_(path1), current_path2_(path2),
      analyzer_(std::make_unique<imgtools::ImageAnalyzer>(path1, path2)) {
  status_ = analyzer_->images_available();
}

auto ImageAnalyzerGui::available() const noexcept -> bool { return status_; }

auto ImageAnalyzerGui::analyzer() noexcept -> imgtools::ImageAnalyzer * {
  return analyzer_.get();
}

auto ImageAnalyzerGui::path1() const noexcept -> const std::filesystem::path & {
  return current_path1_;
}
auto ImageAnalyzerGui::path2() const noexcept -> const std::filesystem::path & {
  return current_path2_;
}

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
  // Load
  image1_ = cv::imread(path1_.string(), cv::IMREAD_UNCHANGED);
  image2_ = cv::imread(path2_.string(), cv::IMREAD_UNCHANGED);

  // Converts image to grayscale to ensure full compatibility in analysis.
  if (!image1_.empty() && !image2_.empty()) {
    // - 1 - channel grayscale
    // - 3 - channel BGR
    // - 4 - channel BGRA
    auto to_gray = [](const cv::Mat &src, cv::Mat &dst) {
      if (src.channels() == 1) {
        dst = src.clone(); // already grayscale
      } else if (src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
      } else if (src.channels() == 4) {
        cv::cvtColor(src, dst, cv::COLOR_BGRA2GRAY);
      } else {
        throw std::runtime_error(
            "Unsupported number of channels for histogram.");
      }
    };

    try {
      to_gray(image1_, grayscale1_);
      to_gray(image2_, grayscale2_);
    } catch (const std::exception &ex) {
      std::println("[Grayscale Conversion]\nError: {}", ex.what());
    }

    return true;
  }

  return false;
}

auto ImageAnalyzer::images() const -> std::pair<cv::Mat, cv::Mat> {
  return {image1_, image2_};
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
  unsigned long size1{0};
  unsigned long size2{0};

  if (std::filesystem::exists(path1_)) {
    size1 = std::filesystem::file_size(path1_);
  }

  if (std::filesystem::exists(path2_)) {
    size2 = std::filesystem::file_size(path2_);
  }

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
  if (grayscale1_.empty() || grayscale2_.empty()) {
    return "[Histogram]\nFailed: One or both images were not converted to "
           "grayscale.\n";
  }

  // --- Step 3: Histogram configuration (1D grayscale) ---
  int histSize = 256;       // bins
  float range[] = {0, 256}; // pixel values
  const float *histRange = range;
  int channel = 0; // grayscale = single channel

  cv::Mat hist1, hist2;

  // --- Step 4: Compute histograms ---
  // cv::calcHist creates a histogram using the selected channels and ranges.
  cv::calcHist(&grayscale1_, 1, &channel, cv::Mat(), hist1, 1, &histSize,
               &histRange, true, false);

  cv::calcHist(&grayscale2_, 1, &channel, cv::Mat(), hist2, 1, &histSize,
               &histRange, true, false);

  // --- Step 5: Normalize for comparability ---
  // Normalization ensures the comparison does not depend on image size.
  cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX);
  cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX);

  // --- Step 6: Compute similarity metrics ---
  // Multiple metrics give more insight about the histogram similarity.
  //
  // Methods:
  //
  // cv::HISTCMP_CORREL - It measures the correlation between the two
  // histograms. Highest value → Highest similarity (1.0 for identical).
  //
  // cv::HISTCMP_CHISQR - Calculates the Chi-square distance.
  // Smaller value → Greater similarity (0.0 for identical histograms).
  //
  // cv::HISTCMP_INTERSECT - Calculates the sum of the minima in each bin
  // of the histograms. Highest value → Highest similarity (equal to the total
  // number of pixels if the histograms are normalized or identical).
  //
  // cv::HISTCMP_BHATTACHARYYA - Measures the distance between the two
  // distributions. Also known as Hellinger Distance.
  // Smaller value → Greatersimilarity (0.0 for identical histograms).
  //
  // cv::HISTCMP_KL_DIV - It measures how different one distribution is from
  // another. Lower value → Greater similarity (0.0 for identical histograms).
  //
  double corr = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
  double chisq = cv::compareHist(hist1, hist2, cv::HISTCMP_CHISQR);
  double inter = cv::compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
  double bhatt = cv::compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
  double kldiv = cv::compareHist(hist1, hist2, cv::HISTCMP_KL_DIV);

  // --- Step 7: Build output string ---
  std::ostringstream oss;
  oss << "[Histogram]\n";
  oss << "Correlation:      " << corr
      << (corr > 0.9    ? "  (IDENTICAL)"
          : corr < -0.9 ? "  (INVERSE)"
          : corr < -0.5 ? "  (ANTI-CORRELATED)"
          : corr < 0.2  ? "  (NO CORRELATION)"
          : corr < 0.5  ? "  (WEAK)"
                        : "  (SIMILAR)")
      << "\n";
  oss << "Chi-Square:       " << chisq << "\n";
  oss << "Intersection:     " << inter << "\n";
  oss << "Bhattacharyya:    " << bhatt << "\n";
  oss << "Kullback-Leibler: " << kldiv << "\n";

  // --- Step 8: Basic interpretation based on correlation value ---
  oss << "Histogram Correlation: " << std::format("{:.4f}", corr) << "\n";

  oss << "Similarity Score: " << std::format("{:.2f}%", corr * 100.0) << "\n";

  oss << "\nInterpretation: ";
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

auto ImageAnalyzer::compare_structural() const -> std::string {

  // --- Step 1: Basic validation ---
  if (image1_.empty() || image2_.empty()) {
    return "[Structural]\nFailed: One or both images are not loaded.\n";
  }

  // --- Step 2: Convert safely to grayscale ---
  if (grayscale1_.empty() || grayscale2_.empty()) {
    return "[Structural]\nFailed: One or both images were not converted to "
           "grayscale.\n";
  }

  // Bkp
  cv::Mat gray1 = grayscale1_;
  cv::Mat gray2 = grayscale2_;

  // --- Step 3: Resize if needed (SSIM requires same size) ---
  if (gray1.size() != gray2.size()) {
    cv::resize(gray1, gray2, gray1.size(), 0, 0, cv::INTER_AREA);
  }

  // --- Step 4: Compute MSE (Mean Squared Error) ---
  // It measures the gross difference between the pixels of the two images.
  cv::Mat diff;
  cv::absdiff(gray1, gray2, diff);
  diff.convertTo(diff, CV_32F);
  diff = diff.mul(diff);

  double mse = cv::sum(diff)[0] / static_cast<double>(grayscale1_.total());

  // --- Step 5: Compute PSNR (Peak Signal-to-Noise Ratio) ---
  // PSNR is directly derived from MSE and measures how "noisy" image 2 is
  // compared to image 1.
  double psnr = (mse == 0.0) ? 99.0 : 10.0 * std::log10((255.0 * 255.0) / mse);

  // --- Step 6: Compute SSIM (Structural Similarity Index) ---
  // It measures perceptual similarity, based on brightness, contrast, and
  // structure.
  auto ssim = [](const cv::Mat &i1, const cv::Mat &i2) -> double {
    const double C1 = 6.5025;
    const double C2 = 58.5225;

    cv::Mat I1, I2;
    i1.convertTo(I1, CV_32F);
    i2.convertTo(I2, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1.mul(I1), sigma1_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2.mul(I2), sigma2_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);

    sigma1_2 -= mu1_2;
    sigma2_2 -= mu2_2;
    sigma12 -= mu1_mu2;

    cv::Mat t1 = (2.0 * mu1_mu2 + C1);
    cv::Mat t2 = (2.0 * sigma12 + C2);
    cv::Mat t3 = (mu1_2 + mu2_2 + C1);
    cv::Mat t4 = (sigma1_2 + sigma2_2 + C2);

    cv::Mat ssim_map;
    cv::divide(t1.mul(t2), t3.mul(t4), ssim_map);

    return cv::mean(ssim_map)[0];
  };

  double ssim_value = ssim(gray1, gray2);

  // --- Step 7: Build output string ---
  std::string mse_quality;
  if (mse == 0.0) {
    mse_quality = "Perfect match (no error)";
  } else if (mse < 10.0) {
    mse_quality = "Very small error (excellent similarity)";
  } else if (mse < 50.0) {
    mse_quality = "Small error (good similarity)";
  } else if (mse < 200.0) {
    mse_quality = "Moderate error (visible differences)";
  } else {
    mse_quality = "High error (images differ strongly)";
  }

  std::string psnr_quality;
  if (!std::isfinite(psnr)) {
    psnr_quality = "Infinite (perfect reconstruction)";
  } else if (psnr > 40.0) {
    psnr_quality = "Excellent (visually identical or extremely similar)";
  } else if (psnr > 30.0) {
    psnr_quality = "Good (small differences)";
  } else if (psnr > 20.0) {
    psnr_quality = "Fair (perceptible degradation)";
  } else {
    psnr_quality = "Poor (noticeable noise or distortion)";
  }

  std::string ssim_quality;
  if (ssim_value > 0.95) {
    ssim_quality = "Excellent structural similarity";
  } else if (ssim_value > 0.80) {
    ssim_quality = "High similarity";
  } else if (ssim_value > 0.60) {
    ssim_quality = "Partial similarity (structure differs)";
  } else if (ssim_value > 0.0) {
    ssim_quality = "Low similarity (different structure)";
  } else {
    ssim_quality = "Possible negative correlation or inversion";
  }

  // Extra detection: strong structural inversion indicator
  bool is_inverse = (ssim_value < 0.0 && mse > 200.0);

  // Formatting output information
  std::ostringstream oss;
  oss << "[Structural]\n";
  oss << "MSE  : " << mse << "   --> " << mse_quality << "\n";
  oss << "PSNR : " << psnr << " dB --> " << psnr_quality << "\n";
  oss << "SSIM : " << ssim_value << "   --> " << ssim_quality << "\n";

  if (is_inverse) {
    oss << "\nDetection: Strong indicators of INVERSION between the "
           "images.\n";
  }

  oss << "\nLegend:\n"
      << " - MSE  (0=perfect)\n"
      << " - PSNR (>40dB excellent)\n"
      << " - SSIM (1.0 perfect, <0 possible inversion)\n";

  // --- Step 8: Interpretation ---
  oss << "\nInterpretation: ";
  if (ssim_value > 0.95)
    oss << "Images are structurally identical";
  else if (ssim_value > 0.75)
    oss << "Strong structural similarity";
  else if (ssim_value > 0.40)
    oss << "Moderate structural similarity";
  else if (ssim_value > 0.10)
    oss << "Weak similarity";
  else
    oss << "Images are structurally different";

  oss << "\n";

  return oss.str();
}

auto imgtools::ImageAnalyzer::compare_features(FeatureMethod method) const
    -> std::string {

  std::ostringstream oss;
  oss << "[Feature Matching]\n";

  // --- Step 1: Validate grayscale conversion ---
  if (grayscale1_.empty() || grayscale2_.empty()) {
    oss << "Error: Images not loaded.\n";
    return oss.str();
  }

  // --- Auxiliary Function : Homography classification ---
  auto classifyHomography = [](const cv::Mat &H) -> std::string {
    if (H.empty())
      return "None";

    double p0 = std::abs(H.at<double>(2, 0));
    double p1 = std::abs(H.at<double>(2, 1));

    if (p0 < 1e-3 && p1 < 1e-3)
      return "Affine/Similarity";
    return "Perspective";
  };

  // --- Auxiliary Function : Quality based on inlier ratio ---
  auto qualityLabel = [](double r) -> std::string {
    if (r > 0.60)
      return "GOOD";
    if (r > 0.30)
      return "MODERATE";
    return "POOR";
  };

  // --- Auxiliary Function : Mean and variance ---
  auto mean = [](const std::vector<double> &v) -> double {
    if (v.empty())
      return 0.0;
    double s = 0.0;
    for (double x : v)
      s += x;
    return s / v.size();
  };

  auto variance = [](const std::vector<double> &v, double m) -> double {
    if (v.size() < 2)
      return 0.0;
    double sum = 0.0;
    for (double x : v)
      sum += (x - m) * (x - m);
    return sum / (v.size() - 1);
  };

  // Step 2: Feature detector/descriptor selection
  cv::Ptr<cv::Feature2D> detector;

  switch (method) {
  case FeatureMethod::ORB:
    detector = cv::ORB::create(1500); // FAST + BRIEF (binary)
    oss << "Method: ORB\n";
    break;

  case FeatureMethod::AKAZE:
    detector = cv::AKAZE::create(); // nonlinear scale space
    oss << "Method: AKAZE\n";
    break;

  case FeatureMethod::SIFT:
    detector = cv::SIFT::create(1200); // gradient-based, float descriptors
    oss << "Method: SIFT\n";
    break;
  }

  bool useHamming =
      (method == FeatureMethod::ORB || method == FeatureMethod::AKAZE);

  cv::Ptr<cv::DescriptorMatcher> matcher =
      useHamming
          ? cv::DescriptorMatcher::create(
                cv::DescriptorMatcher::BRUTEFORCE_HAMMING)
          : cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

  std::vector<cv::KeyPoint> kp1, kp2;
  cv::Mat desc1, desc2;

  detector->detectAndCompute(grayscale1_, cv::noArray(), kp1, desc1);
  detector->detectAndCompute(grayscale2_, cv::noArray(), kp2, desc2);

  oss << "Keypoints Image 1: " << kp1.size() << "\n";
  oss << "Keypoints Image 2: " << kp2.size() << "\n";

  if (kp1.empty() || kp2.empty() || desc1.empty() || desc2.empty()) {
    oss << "Error: Could not compute descriptors.\n";
    return oss.str();
  }

  // Step 3: KNN matching + Lowe ratio test
  std::vector<std::vector<cv::DMatch>> knnMatches;
  matcher->knnMatch(desc1, desc2, knnMatches, 2);

  std::vector<cv::DMatch> goodMatches;
  std::vector<double> ratios;
  std::vector<double> distances;

  const float ratioThresh = 0.75f;

  for (auto &pair : knnMatches) {
    if (pair.size() < 2)
      continue;

    float r = pair[0].distance / pair[1].distance;
    if (r < ratioThresh) {
      goodMatches.push_back(pair[0]);
      ratios.push_back(r);
      distances.push_back(pair[0].distance);
    }
  }

  oss << "Good Matches: " << goodMatches.size() << "\n";

  if (goodMatches.size() < 4) {
    oss << "Not enough matches for homography.\n";
    return oss.str();
  }

  // Step 4: Compute Homography with RANSAC
  std::vector<cv::Point2f> pts1, pts2;
  for (auto &m : goodMatches) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
  }

  std::vector<unsigned char> inlierMask;
  cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, inlierMask);

  int inliers = std::count(inlierMask.begin(), inlierMask.end(), 1);
  int outliers = goodMatches.size() - inliers;
  double inlierRatio = (double)inliers / goodMatches.size();

  // Step 5: Report RANSAC results
  oss << "\n[RANSAC]\n";
  oss << "Inliers: " << inliers << "\n";
  oss << "Outliers: " << outliers << "\n";
  oss << "Inlier Ratio: " << inlierRatio << " (" << qualityLabel(inlierRatio)
      << ")\n";

  // Step 6: Homography details
  oss << "\n[Homography]\n";
  if (!H.empty()) {
    oss << "Status: FOUND\n";
    oss << "Type: " << classifyHomography(H) << "\n";
    oss << "Determinant: " << std::abs(cv::determinant(H)) << "\n";
  } else {
    oss << "Status: NOT FOUND\n";
  }

  // Step 7: Match confidence metrics
  double meanDist = mean(distances);
  double varDist = variance(distances, meanDist);
  double meanRatio = mean(ratios);

  oss << "\n[Match Confidence]\n";
  oss << "Average Match Distance: " << meanDist << "\n";
  oss << "Distance Variance: " << varDist << "\n";
  oss << "Mean Lowe Ratio: " << meanRatio << "\n";

  // Step 8: High-level interpretation
  oss << "\n[Summary]\n";
  if (inlierRatio > 0.60)
    oss << "Images have STRONG structural similarity.\n";
  else if (inlierRatio > 0.30)
    oss << "Images have MODERATE similarity.\n";
  else
    oss << "Images are likely DIFFERENT.\n";

  return oss.str();
}

// Export a detailed report with headers and separators.
auto ImageAnalyzer::export_report(const std::filesystem::path &output_path,
                                  FeatureMethod method) const -> bool {

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
  oss << compare_structural() << "\n";
  oss << compare_features(method) << "\n";

  // Footer
  oss << "----------------------------------------\n";
  oss << "End of report.\n";

  file << oss.str();
  file.close();

  return true;
}

auto ImageAnalyzer::export_report(std::string_view output_path,
                                  FeatureMethod method) -> bool {
  return export_report(std::filesystem::path(output_path), method);
}

} // namespace imgtools

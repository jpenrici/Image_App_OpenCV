---

# Image_App_OpenCV

Simple image analyzer using C++ and OpenCV.

---

## Overview

A Qt-based graphical application that loads two images and performs simple analyses using OpenCV.

Analysis:

- Basic metadata comparison
- Color-space analysis
- Histogram similarity
- Structural similarity (MSE, PSNR, SSIM)
- Feature-based matching (ORB, AKAZE, SIFT)

---

## Requirements

* **C++ Standard:** C++23 or newer
* **CMake:** Version 3.25 or higher
* **Compiler:** GCC 15 or Clang 18+

* **OpenCV:** Version 4.10 or higher
  
  * [OpenCV](https://opencv.org/releases/)

* **Qt:** Version 6

  * [Qt 6 Widgets](https://doc.qt.io/qt-6/qtwidgets-index.html)

---

### Install dependencies (Ubuntu/Debian)

```bash
sudo apt install qt6-base-dev libopencv-dev
```

---

## Build and Run Instructions

```bash
# Clone repository
git clone <repository_url>
cd Image_App

# Configure and build
cmake -B build
cmake --build build

# Run - Qt
cd build/bin
./imageTools_qt6

# Run - Library test
cd build/bin
./imagetools_test
```

---

## References

* [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
* [LearnOpenCV](https://learnopencv.com/getting-started-with-opencv/)

---

## License

This project is distributed for educational and research purposes under the **MIT License**.

---


## Display



![display](https://github.com/jpenrici/Image_App_OpenCV/blob/main/display/display.png)


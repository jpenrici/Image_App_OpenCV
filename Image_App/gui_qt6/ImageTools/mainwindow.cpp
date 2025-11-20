#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QTabWidget>

#include <print>
#include <sstream>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);

  ui->tableReport->setColumnCount(2);
  ui->tableReport->setHorizontalHeaderLabels({"Metric", "Value"});
  ui->tableReport->horizontalHeader()->setStretchLastSection(true);

  // Load buttons
  connect(ui->btnLoadImg1, &QPushButton::clicked, this,
          [this]() { loadImage1(); });

  connect(ui->btnLoadImg2, &QPushButton::clicked, this,
          [this]() { loadImage2(); });

  // Visualization buttons
  connect(ui->btnShowHist, &QPushButton::clicked, this,
          [this] { showHistograms(); });

  connect(ui->btnShowFeatures, &QPushButton::clicked, this,
          [this] { showFeatures(); });

  connect(ui->btnExportReport, &QPushButton::clicked, this,
          [this] { exportReport(); });
}

MainWindow::~MainWindow() { delete ui; }

auto MainWindow::loadImage() -> QString {
  // Extensões mais comuns suportadas pelo OpenCV
  const QString filter =
      "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.ppm *.pgm *.gif)";

  QString p =
      QFileDialog::getOpenFileName(this, "Select Image", QString(), filter);

  if (p.isEmpty()) {
    QMessageBox::warning(this, "Invalid Image", "Invalid name!");
    return {};
  }

  return p;
}

void MainWindow::loadImage1() {
  path1_ = loadImage();
  if (!path1_.isEmpty()) {
    updatePreview();
    tryRebuildAnalyzer();
  }
}

void MainWindow::loadImage2() {
  path2_ = loadImage();
  if (!path2_.isEmpty()) {
    updatePreview();
    tryRebuildAnalyzer();
  }
}

void MainWindow::updatePreview() {
  if (!path1_.isEmpty()) {
    QPixmap pm(path1_);
    ui->labelImg1->setPixmap(pm.scaled(280, 280, Qt::KeepAspectRatio));
  }

  if (!path2_.isEmpty()) {
    QPixmap pm(path2_);
    ui->labelImg2->setPixmap(pm.scaled(280, 280, Qt::KeepAspectRatio));
  }
}

void MainWindow::tryRebuildAnalyzer() {
  if (path1_.isEmpty() || path2_.isEmpty())
    return;

  // Convert QString to std::filesystem::path
  std::string p1 = path1_.toStdString();
  std::string p2 = path2_.toStdString();

  // Check if analyzer already valid AND same images
  if (analyzer_) {
    if (analyzer_->path1() == p1 && analyzer_->path2() == p2) {
      return; // paths unchanged
    }
  }

  analyzer_ = std::make_unique<ImageAnalyzerGui>(p1, p2);

  if (!analyzer_->available()) {
    QMessageBox::warning(this, "Error", "Could not load one or both images.");
  }

  fillReport();
}

void MainWindow::fillReport() {
  if (!analyzer_ || !analyzer_->available()) {
    ui->tableReport->clearContents();
    return;
  }

  auto a = analyzer_->analyzer();

  // combine all analyses into key-value lines
  std::vector<std::pair<std::string, std::string>> rows;

  auto addLines = [&](const std::string &label, const std::string &txt) {
    rows.push_back({label, ""});
    std::istringstream iss(txt);
    std::string line;
    while (std::getline(iss, line))
      if (!line.empty())
        rows.push_back({"", line});
  };

  addLines("Basic", a->compare_basic());
  addLines("Color Space", a->compare_color_space());
  addLines("Histogram", a->compare_histogram());
  addLines("Structural", a->compare_structural());
  addLines("Features", a->compare_features(imgtools::FeatureMethod::AKAZE));

  ui->tableReport->setRowCount(int(rows.size()));

  for (int r = 0; r < (int)rows.size(); r++) {
    ui->tableReport->setItem(
        r, 0, new QTableWidgetItem(QString::fromStdString(rows[r].first)));
    ui->tableReport->setItem(
        r, 1, new QTableWidgetItem(QString::fromStdString(rows[r].second)));
  }
}

void MainWindow::showHistograms() {
  if (!analyzer_ || !analyzer_->available())
    return;
  analyzer_->analyzer()->show_histograms();
}

void MainWindow::showFeatures() {
  if (!analyzer_ || !analyzer_->available())
    return;
  analyzer_->analyzer()->show_features(imgtools::FeatureMethod::AKAZE);
}

void MainWindow::exportReport() {

  if (!analyzer_ || !analyzer_->available()) {
    QMessageBox::warning(this, "Warning",
                         "Load two valid images before exporting a report.");
    return;
  }

  auto name1 = std::filesystem::path(path1_.toStdString()).filename().string();
  auto name2 = std::filesystem::path(path2_.toStdString()).filename().string();

  auto output_name =
      std::format("report_{}_{}", name1.substr(0, name1.find('.')),
                  name2.substr(0, name2.find('.')));
  auto output_path = std::format("./{}", output_name);

  auto suggested = QString::fromStdString(output_name + ".txt");

  auto p = QFileDialog::getSaveFileName(this, "Save As", suggested,
                                        "Text Files (*.txt);;All Files (*.*)");

  if (!p.isEmpty()) {
    auto path = p;
    bool ok = analyzer_->analyzer()->export_report(
        path.toStdString()); // Use the standard method
    if (ok) {
      QMessageBox::information(this, "Success",
                               "Report exported successfully:\n" + path);
    } else {
      QMessageBox::critical(this, "Error", "Failed to export the report.");
    }
  }
}

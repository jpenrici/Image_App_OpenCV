#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QString>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);

  // Connect buttons using lambdas (recommended by Qt6)
  connect(ui->btnLoadImg1, &QPushButton::clicked, this,
          [this]() { loadImage1(); });

  connect(ui->btnLoadImg2, &QPushButton::clicked, this,
          [this]() { loadImage2(); });

  connect(ui->btnBasic, &QPushButton::clicked, this, [this]() { runBasic(); });

  connect(ui->btnHistogram, &QPushButton::clicked, this,
          [this]() { runHistogram(); });

  connect(ui->btnStructural, &QPushButton::clicked, this,
          [this]() { runStructural(); });

  connect(ui->btnFeatures, &QPushButton::clicked, this,
          [this]() { runFeatures(); });
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::loadImage1() {
  auto p = QFileDialog::getOpenFileName(this, "Select Image 1");
  if (!p.isEmpty()) {
    path1_ = p;
    updatePreview();
    tryRebuildAnalyzer();
  }
}

void MainWindow::loadImage2() {
  auto p = QFileDialog::getOpenFileName(this, "Select Image 2");
  if (!p.isEmpty()) {
    path2_ = p;
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
}

void MainWindow::runBasic() {
  if (!analyzer_ || !analyzer_->available()) {
    ui->txtReport->setPlainText("Images not available.");
    return;
  }

  auto ia = analyzer_->analyzer();

  QString report;
  report += QString::fromStdString(ia->compare_basic()) + "\n";
  report += QString::fromStdString(ia->compare_color_space());

  ui->txtReport->setPlainText(report);
}

void MainWindow::runHistogram() {
  if (!analyzer_ || !analyzer_->available()) {
    ui->txtReport->setPlainText("Images not available.");
    return;
  }

  auto ia = analyzer_->analyzer();

  QString report = QString::fromStdString(ia->compare_histogram());
  ui->txtReport->setPlainText(report);
}

void MainWindow::runStructural() {
  if (!analyzer_ || !analyzer_->available()) {
    ui->txtReport->setPlainText("Images not available.");
    return;
  }

  auto ia = analyzer_->analyzer();

  QString report = QString::fromStdString(ia->compare_structural());
  ui->txtReport->setPlainText(report);
}

void MainWindow::runFeatures() {
  if (!analyzer_ || !analyzer_->available()) {
    ui->txtReport->setPlainText("Images not available.");
    return;
  }

  auto ia = analyzer_->analyzer();

  QString report = QString::fromStdString(
      ia->compare_features(imgtools::FeatureMethod::AKAZE));

  ui->txtReport->setPlainText(report);
}

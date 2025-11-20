#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>

#include "imagetoolsgui.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

/**
 * @brief The MainWindow class
 * Graphical interface for the imgtools library.
 */
class MainWindow : public QMainWindow {

  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private:
  Ui::MainWindow *ui;

  QString path1_;
  QString path2_;

  std::unique_ptr<ImageAnalyzerGui> analyzer_;

  // Internal helpers
  void loadImage1();
  void loadImage2();

  void updatePreview();
  void fillReport();
  void tryRebuildAnalyzer();

  // Visualizations (OpenCV)
  void showHistograms();
  void showFeatures();
};

#endif // MAINWINDOW_H

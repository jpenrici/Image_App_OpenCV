#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QString>
#include <QTextEdit>

#include <expected>

#include "imagetoolsgui.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

/**
 * @brief The MainWindow class
 * Graphical interface for the imgtools library.
 *
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

  std::unique_ptr<ImageAnalyzerGui> analyzerGui_;

  void loadImage1();
  void loadImage2();
  void tryCreateAnalyzer();

  void displayImage(const cv::Mat &img, QLabel *target);
  void appendLog(const QString &msg);
};

#endif // MAINWINDOW_H

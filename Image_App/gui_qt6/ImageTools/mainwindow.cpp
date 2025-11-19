#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::loadImage1()
{

}

void MainWindow::loadImage2()
{

}

void MainWindow::tryCreateAnalyzer()
{

}

void MainWindow::displayImage(const cv::Mat &img, QLabel *target)
{

}

void MainWindow::appendLog(const QString &msg)
{

}

#include "bookshelf_rviz_image_panel/debug_image_panel.hpp"

#include <algorithm>
#include <string>

#include <QLabel>
#include <QMetaObject>
#include <QPixmap>
#include <QResizeEvent>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>

namespace bookshelf_rviz_image_panel
{

DebugImagePanel::DebugImagePanel(QWidget * parent)
: rviz_common::Panel(parent), image_label_(new QLabel("Waiting for /slot_detector/debug_image", this))
{
  image_label_->setAlignment(Qt::AlignCenter);
  image_label_->setMinimumSize(320, 240);
  image_label_->setStyleSheet("QLabel { background: #202124; color: #d0d0d0; }");

  auto * layout = new QVBoxLayout(this);
  layout->setContentsMargins(2, 2, 2, 2);
  layout->addWidget(image_label_);
}

void DebugImagePanel::onInitialize()
{
  auto abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!abstraction) {
    image_label_->setText("RViz ROS node unavailable");
    return;
  }

  node_ = abstraction->get_raw_node();
  subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
    "/slot_detector/debug_image", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::Image::ConstSharedPtr message) {handleImage(message);});
  RCLCPP_INFO(node_->get_logger(), "Qt debug-image panel subscribed to /slot_detector/debug_image");
}

void DebugImagePanel::handleImage(sensor_msgs::msg::Image::ConstSharedPtr message)
{
  QImage image;
  const auto * data = message->data.data();
  const int width = static_cast<int>(message->width);
  const int height = static_cast<int>(message->height);
  const int step = static_cast<int>(message->step);

  if (message->encoding == "rgb8") {
    image = QImage(data, width, height, step, QImage::Format_RGB888).copy();
  } else if (message->encoding == "bgr8") {
    image = QImage(data, width, height, step, QImage::Format_RGB888).rgbSwapped();
  } else if (message->encoding == "rgba8") {
    image = QImage(data, width, height, step, QImage::Format_RGBA8888).copy();
  } else if (message->encoding == "bgra8") {
    image = QImage(data, width, height, step, QImage::Format_ARGB32).rgbSwapped();
  } else if (message->encoding == "mono8") {
    image = QImage(data, width, height, step, QImage::Format_Grayscale8).copy();
  } else {
    return;
  }

  if (!first_image_logged_) {
    first_image_logged_ = true;
    RCLCPP_INFO(
      node_->get_logger(), "Qt debug-image panel received %ux%u %s",
      message->width, message->height, message->encoding.c_str());
  }

  QMetaObject::invokeMethod(
    this, [this, image]() {showImage(image);}, Qt::QueuedConnection);
}

void DebugImagePanel::showImage(const QImage & image)
{
  latest_image_ = image;
  image_label_->setPixmap(
    QPixmap::fromImage(latest_image_).scaled(
      image_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void DebugImagePanel::resizeEvent(QResizeEvent * event)
{
  rviz_common::Panel::resizeEvent(event);
  if (!latest_image_.isNull()) {
    showImage(latest_image_);
  }
}

}  // namespace bookshelf_rviz_image_panel

PLUGINLIB_EXPORT_CLASS(bookshelf_rviz_image_panel::DebugImagePanel, rviz_common::Panel)

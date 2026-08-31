#ifndef BOOKSHELF_RVIZ_IMAGE_PANEL__DEBUG_IMAGE_PANEL_HPP_
#define BOOKSHELF_RVIZ_IMAGE_PANEL__DEBUG_IMAGE_PANEL_HPP_

#include <memory>

#include <QImage>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <sensor_msgs/msg/image.hpp>

class QLabel;
class QResizeEvent;

namespace bookshelf_rviz_image_panel
{

class DebugImagePanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit DebugImagePanel(QWidget * parent = nullptr);
  void onInitialize() override;

protected:
  void resizeEvent(QResizeEvent * event) override;

private:
  void handleImage(sensor_msgs::msg::Image::ConstSharedPtr message);
  void showImage(const QImage & image);

  QLabel * image_label_;
  QImage latest_image_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  bool first_image_logged_{false};
};

}  // namespace bookshelf_rviz_image_panel

#endif  // BOOKSHELF_RVIZ_IMAGE_PANEL__DEBUG_IMAGE_PANEL_HPP_

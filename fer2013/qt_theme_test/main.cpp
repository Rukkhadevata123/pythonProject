#include <QApplication>
#include <QPushButton>
#include <QStyleFactory>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QWidget>
#include <QLineEdit>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // 设置样式为 Kvantum
    app.setStyle("kvantum");

    // 创建主窗口
    QWidget window;
    window.setWindowTitle("Kvantum 主题测试");
    window.resize(300, 200);

    // 创建布局
    QVBoxLayout *layout = new QVBoxLayout(&window);

    // 创建文本框
    QLineEdit *lineEdit = new QLineEdit(&window);
    lineEdit->setPlaceholderText("输入一些文本...");
    layout->addWidget(lineEdit);

    // 创建按钮
    QPushButton *button1 = new QPushButton("按钮 1", &window);
    QPushButton *button2 = new QPushButton("按钮 2", &window);
    QPushButton *button3 = new QPushButton("按钮 3", &window);

    // 添加按钮到布局
    layout->addWidget(button1);
    layout->addWidget(button2);
    layout->addWidget(button3);

    // 连接按钮点击事件
    QObject::connect(button1, &QPushButton::clicked, [&]() {
        QMessageBox::information(&window, "信息", "按钮 1 被点击！");
    });

    QObject::connect(button2, &QPushButton::clicked, [&]() {
        QMessageBox::information(&window, "信息", "按钮 2 被点击！");
    });

    QObject::connect(button3, &QPushButton::clicked, [&]() {
        QMessageBox::information(&window, "信息", "按钮 3 被点击！");
    });

    // 显示窗口
    window.show();
    return app.exec();
}
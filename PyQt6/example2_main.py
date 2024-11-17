import sys
import warnings

import requests
from PyQt6.QtWidgets import QApplication, QDialog, QMainWindow

from example2 import Ui_Dialog

warnings.filterwarnings("ignore", category=DeprecationWarning)

class MyDialog(QDialog, QMainWindow):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("城市天气预报")

    def query(self):
        city = self.ui.comboBox.currentText()
        cityCode = self.getCode(city)
        r = requests.get(
            "https://restapi.amap.com/v3/weather/weatherInfo?key=def944d538b9cf8ad1ee992fcf6cb7e1&city={}".format(
                cityCode)
        )
        if r.status_code == 200:
            data = r.json()["lives"][0]
            weatherMsg = "城市：{}\n天气：{}\n温度：{}°C\n风向：{}\n风力：{}\n湿度：{}\n发布时间: {}".format(
                data["city"], data["weather"], data["temperature"], data["winddirection"], data["windpower"],
                data["humidity"], data["reporttime"])
        else :
            weatherMsg = "查询失败"
        self.ui.textEdit.setText(weatherMsg)

    def getCode(self, cityName):
        cityDict = {"北京": "110000",
                    "苏州": "320500",
                    "上海": "310000"}
        return cityDict.get(cityName, '101010100')

    def clear(self):
        self.ui.textEdit.clear()
        self.ui.comboBox.setCurrentIndex(0)
        self.ui.comboBox.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyDialog()
    window.show()
    sys.exit(app.exec())
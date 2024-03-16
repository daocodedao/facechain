
import requests
import socket
from  utils.util import Util
import json

def getNetworkIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0))
    return s.getsockname()[0]

serverUrl = "http://39.105.194.16:9191/v1/chat/completions/"
# localIp = getNetworkIp()
# print(f"localIp:{localIp}")
# if "192.168.0" in localIp:
#     serverUrl = "http://192.168.0.67:9191/v1/chat/completions/"

def translate_en_to_zh(inSrc):
    # url = "http://39.105.194.16:9191/v1/chat/completions/"  # 替换为您要发送请求的URL
    data = {
            "systemRole": "system",
            "systemContent": "You are a helpful assistant.",
            "userRole":"role",
            "userContent":"Trump was always bothered by how Trump Tower fell 41 feet short of the General Motors building two blocks north."
    }

    systemContent = f"你是一个翻译助手，将后续的英文翻译成中文。不要输出额外的内容。"
    userContent = f"翻译以下内容：\n{inSrc}"
    # data["systemContent"] = systemContent
    data["systemContent"] = systemContent
    data["userContent"] = userContent

    response = requests.post(serverUrl, json=data)

    if response.status_code == 200:
        # api_logger.info("请求成功")
        retJson = response.json()
        # api_logger.info(retJson)
        ret_text = retJson["message"]
        return ret_text
    else:
        # api_logger.info("请求失败，状态码：", response.status_code)
        # api_logger.info(response.text)
        return ""

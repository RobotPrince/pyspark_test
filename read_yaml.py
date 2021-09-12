import yaml
import os

def config_test(filename):
    with open(filename, 'r') as f:
        return yaml.load(f)


# filename = './config.yaml'
# 获取当前脚本所在文件夹路径
curPath = os.path.dirname(os.path.realpath(__file__))
# 获取yaml文件路径
cfgPath = os.path.join(curPath, "config.yaml")
path_map = config_test(cfgPath)
print(path_map)
print(type(path_map))
source = path_map.get("source")
oscar = source.get("oscar")

print(oscar)



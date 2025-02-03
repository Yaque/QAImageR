from PIL import Image, ImageDraw, ImageOps
import random
import os

def add_random_border(image_path, output_path):
    # 打开图像
    image = Image.open(image_path)
    # 生成随机的边框宽度，范围在 10 到 50 像素之间
    border_width = random.randint(0, 50)
    # 生成随机的边框颜色，使用 RGB 模式
    border_color = (255, 255, 255)
    # 扩展图像，添加边框
    bordered_image = ImageOps.expand(image, border=border_width, fill=border_color)
    # 保存结果图像
    bordered_image.save(output_path)



# 单纯批量改变图片尺寸调整

# 定义要遍历的文件夹路径
folder_path = './images/Test'

# 遍历文件夹中的所有文件和子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 获取文件的完整路径
        file_path = os.path.join(root, file)
        add_random_border(file_path, "images/TH/" + file.split('.')[0] + "q.png")
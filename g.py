import os

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sign = ['+', '-', 'x']


#width=180
#height=60

def random_captcha_text(captcha_size=5):
    captcha_text = []
    c = random.choice(number)
    captcha_text.append(c)
    c = random.choice(sign)
    captcha_text.append(c)
    c = random.choice(number)
    captcha_text.append(c)
    captcha_text.append('=')
    captcha_text.append('?')
    return captcha_text

from PIL import Image, ImageDraw, ImageFont
import random

def generate_captcha_image(captcha_text):

    width, height = 120, 40

    image = Image.new('RGB', (width, height), (255, 255, 255))
    font = ImageFont.truetype('1.ttf', 35)

    draw = ImageDraw.Draw(image)
    
    # 定义要画的小圆圈的数量和大小
    # number_of_circles = random.randrange(2, 3)
    number_of_circles = 2

    # 在随机位置画小圆圈
    for _ in range(number_of_circles):
        circle_diameter = random.randrange(7, 15)
        # 随机选择圆心的位置，确保圆不会超出图像边界
        x = random.randint(0, width - circle_diameter)
        y = random.randint(0, height - circle_diameter)
        
        # 定义圆的边界框
        bbox = (x, y, x + circle_diameter, y + circle_diameter)
        
        # 画圆，可以指定颜色
        draw.ellipse(bbox, outline=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

    for i, c in enumerate(captcha_text):
        draw.text((i*random.randrange(20, 25)+6, 7), c, font=font, fill=(random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)))

    return image


    
def gen_captcha_text_and_image(total_number,flag):
    for i in range(total_number):
        captcha_text = random_captcha_text()
        captcha_text = ''.join(captcha_text)
        generate_captcha_image(captcha_text).save('./images/' + flag + '/' + captcha_text[:-2] + str(i) + '.png', 'png')


if __name__ == "__main__":
    if not os.path.exists('images/Train'):
        os.mkdir('images/Train')
    if not os.path.exists('images/Test'):
        os.mkdir('images/Test')

    gen_captcha_text_and_image(500, 'Train')
    gen_captcha_text_and_image(50, 'Test')
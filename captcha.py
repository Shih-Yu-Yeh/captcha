import requests
import cv2

# 獲取驗證碼圖像的URL
captcha_url = "https://www.ezfunds.com.tw/CreateAccount/Draw?isLogIn=True"

# 下載驗證碼圖像並保存到本地
response_captcha = requests.get(captcha_url)
with open("captcha.jpg", "wb") as f:
    f.write(response_captcha.content)

# 讀取驗證碼圖像
image = cv2.imread("captcha.jpg", cv2.IMREAD_GRAYSCALE)

# 檢查圖像是否成功讀取
if image is None:
    print(f"Failed to read image from {captcha_url}")
else:
    # 進行閾值處理，將圖像轉換為二進制圖像
    _, binary_image_otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 尋找輪廓
    contours, hierarchy = cv2.findContours(
        binary_image_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 繪製輪廓
    image_with_contours = cv2.drawContours(
        image.copy(), contours, -1, (0, 0, 255), 3)

    # 打印轮廓的数量
    print(f"Found {len(contours)} contours")

    # 創建一個空列表，用於存儲每個字符的邊界框
    bounding_boxes = []

    # 定義閾值，根據長和寬篩選字符
    min_width = 10  # 最小寬度閾值
    min_height = 10  # 最小高度閾值
    max_size = 40  # 最大尺寸閾值

    # 遍歷每個輪廓，計算其邊界框並添加到列表中
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_width and h >= min_height and (w <= max_size and h <= max_size):
            bounding_boxes.append((x, y, w, h))

    # 根據邊界框的x坐標進行排序，以便按照從左到右的順序分割字符
    bounding_boxes.sort(key=lambda x: x[0])

    # 創建一個空列表，用於存儲每個字符的圖像
    characters = []

    # 遍歷每個邊界框，根據其坐標和寬高在二進制圖像上截取相應的區域，並添加到列表中
    for x, y, w, h in bounding_boxes:
        character = binary_image_otsu[y:y+h, x:x+w]
        characters.append(character)

    # 打印分割出的字符數量
    print(f"Segmented {len(bounding_boxes)} characters")

    # 顯示原始圖像和處理後的圖像
   # cv2.imshow("Original image", image)
    # cv2.imshow("Binary image with Otsu's method", binary_image_otsu)
    # cv2.imshow("Image with contours", image_with_contours)

    # 创建一个字符示例文件名与字符的映射字典
    char_mapping = {
        "template0.jpg": "0",
        "template1.jpg": "1",
        "template2.jpg": "2",
        "template3.jpg": "3",
        "template4.jpg": "4",
        "template5.jpg": "5",
        "template6.jpg": "6",
        "template7.jpg": "7",
        "template8.jpg": "8",
        "template9.jpg": "9"
        # 添加其他字符示例和对应的字符
    }

    # 遍历每个字符图像，识别并输出相应的字符标签
    for i, character_image in enumerate(characters):
        recognized_char = None

        # 在这里添加字符识别的代码，可以使用模板匹配或其他方法
        # 以下是一个示例使用模板匹配的方法，你可以根据需要进行修改
        for template_file, char_label in char_mapping.items():
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)

            if template is not None and template.shape[0] <= character_image.shape[0] and template.shape[1] <= character_image.shape[1]:
                corr = cv2.matchTemplate(
                    character_image, template, cv2.TM_CCORR_NORMED)
                _, confidence, _, _ = cv2.minMaxLoc(corr)
                if confidence > 0.86:  # 根据实际情况调整阈值
                    recognized_char = char_label
                    break

        if recognized_char is not None:
            print(f"Character {i + 1}: {recognized_char}")
        else:
            print(f"Character {i + 1}: Not recognized")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

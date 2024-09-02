def get_image_base64(image_path):
    import base64

    # 打开图片文件
    with open(image_path, "rb") as image_file:
        img_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    return img_base64


def get_image_base64_url(image_path):
    img_base64 = get_image_base64(image_path)

    return f"data:image/jpeg;base64,{img_base64}"


def is_image_node(file_type: str) -> bool:
    image_types = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp"]
    return file_type.lower() in image_types

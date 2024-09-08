import os
import time

import fitz
import torch
from PIL import Image
from transformers import AutoModelForObjectDetection

from utils.log_utils import LogUtils


def pdf_to_images(pdf_file):
    """将 PDF 每页转成一个 PNG 图像"""
    # 保存路径为原 PDF 文件名（不含扩展名）
    output_directory_path, _ = os.path.splitext(pdf_file)

    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    # 加载 PDF 文件
    pdf_document = fitz.open(pdf_file)

    # 每页转一张图
    for page_number in range(pdf_document.page_count):
        # 取一页
        page = pdf_document[page_number]

        # 转图像
        pix = page.get_pixmap()

        # 从位图创建 PNG 对象
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 保存 PNG 文件
        image.save(f"./{output_directory_path}/page_{page_number + 1}.png")

    # 关闭 PDF 文件
    pdf_document.close()
    return output_directory_path


class MaxResize(object):
    """缩放图像"""

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


class ExtractPdfTable:
    def _init_detection_transform(self):
        import torchvision.transforms as transforms
        # 图像预处理
        self.detection_transform = transforms.Compose(
            [
                MaxResize(800),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __init__(self, table_transformer_path: str = None):
        # 加载 TableTransformer 模型
        if table_transformer_path is None:
            table_transformer_path = "F:/HuggingFace/Other/table-transformer-detection"
        start_time = time.time()
        self.model = AutoModelForObjectDetection.from_pretrained(table_transformer_path)
        LogUtils.log_info("TableTransformer 模型加载完成，耗时：{}s".format(time.time() - start_time))

        self._init_detection_transform()

    # 识别后的坐标换算与后处理

    def _outputs_to_objects(self, outputs, img_size, id2label):
        """从模型输出中取定位框坐标"""

        def box_cxcywh_to_xyxy(x):
            """坐标转换"""
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)

        def rescale_bboxes(out_bbox, size):
            """区域缩放"""
            width, height = size
            boxes = box_cxcywh_to_xyxy(out_bbox)
            boxes = boxes * torch.tensor(
                [width, height, width, height], dtype=torch.float32
            )
            return boxes

        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
        pred_bboxes = [
            elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)
        ]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == "no object":
                objects.append(
                    {
                        "label": class_label,
                        "score": float(score),
                        "bbox": [float(elem) for elem in bbox],
                    }
                )

        return objects

    def detect_and_crop_save_table(self, pad_image_path: str):
        """识别表格，并将表格部分单独存为图像文件"""

        # 加载图像（PDF页）
        image = Image.open(pad_image_path)

        filename, _ = os.path.splitext(os.path.basename(pad_image_path))

        # 输出路径
        cropped_table_directory = os.path.join(os.path.dirname(pad_image_path), "table_images")

        if not os.path.exists(cropped_table_directory):
            os.makedirs(cropped_table_directory)

        # 预处理
        pixel_values = self.detection_transform(image).unsqueeze(0)

        # 识别表格
        with torch.no_grad():
            outputs = self.model(pixel_values)

        # 后处理，得到表格子区域
        id2label = self.model.config.id2label
        id2label[len(self.model.config.id2label)] = "no object"
        detected_tables = self._outputs_to_objects(outputs, image.size, id2label)

        LogUtils.log_info(f"number of tables detected {len(detected_tables)}")

        for idx in range(len(detected_tables)):
            # 将识别从的表格区域单独存为图像
            cropped_table = image.crop(detected_tables[idx]["bbox"])
            cropped_table.save(os.path.join(cropped_table_directory, f"{filename}_{idx}.png"))

    def extract(self, pdf_file):
        output_directory_path = pdf_to_images(pdf_file)

        image_files = os.listdir(output_directory_path)
        total_images_len = len(image_files)

        print(f"Total images to process: {total_images_len}")

        for index, filename in enumerate(image_files):
            # 生成完整路径
            img_path = os.path.join(output_directory_path, filename)
            LogUtils.log_info(f"Processing image {index + 1}/{total_images_len}: {filename}")

            self.detect_and_crop_save_table(img_path)


if __name__ == "__main__":
    extract = ExtractPdfTable()

    # pdf_path = "F:/AiData/zh/103：大模型应用开发极简入门：基于 GPT-4 和 ChatGPT_2024.pdf"
    # pdf_path = "F:/AiData/table/attention_is_all_you_need.pdf"
    # pdf_path = "F:/AiData/table/国家人工智能产业综合标准化体系建设指南（2024版）.pdf"
    pdf_path = "F:/AiData/table/llama2_page8.pdf"
    extract.extract(pdf_path)

    # img_path="F:/AiData/zh/103：大模型应用开发极简入门：基于 GPT-4 和 ChatGPT_2024/aTest/page_21.png"
    # extract.detect_and_crop_save_table(img_path)

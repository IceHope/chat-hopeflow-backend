import os
import time
from typing import Optional, List
import uuid

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

from rag.reader.image_reader import HopeImageVisionLLMReader
from rag.reader.pdf.extract_pdf_img import parse_pdf_to_images
from utils.log_utils import LogUtils

from tqdm import tqdm


class ReaderManager:
    def __init__(self):
        # 定义支持的文件类型及其对应的读取器
        self.file_extractor = {
            ".pdf": PyMuPDFReader(),
            ".png": HopeImageVisionLLMReader(),
            ".jpg": HopeImageVisionLLMReader(),
            ".jpeg": HopeImageVisionLLMReader(),
        }

    def manual_load_pdf(
            self, input_dir: Optional[str] = None, input_files: Optional[List] = None
    ) -> None:
        """
        手动加载PDF文件并提取图片

        Args:
            input_dir: 输入目录路径
            input_files: 输入文件列表
        """
        LogUtils.log_info("开始手动加载PDF文件")

        if not input_dir and not input_files:
            raise ValueError("必须提供 `input_dir` 或 `input_files` 中的一个。")

        # 列出PDF文件
        if input_dir:
            pdf_files = [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith(".pdf")
            ]
            LogUtils.log_info(f"在 {input_dir} 目录下找到 {len(pdf_files)} 个PDF文件")
        elif input_files:
            pdf_files = [f for f in input_files if f.lower().endswith(".pdf")]
            LogUtils.log_info(f"在提供的文件列表中找到 {len(pdf_files)} 个PDF文件")

        if pdf_files:
            for pdf_file in tqdm(pdf_files, desc="提取PDF文件的图片"):
                LogUtils.log_info(f"正在处理文件: {pdf_file}")
                parse_pdf_to_images(pdf_file)

        LogUtils.log_info("PDF文件图片提取完成")

    def _extract_pdf_image_document(
            self, pdf_file_paths: list[str], need_extrac_img: bool = True
    ) -> list[Document]:
        all_img_documents = []

        for pdf_file_path in tqdm(pdf_file_paths, desc="提取PDF文件的图片文档"):
            output_dir, _ = os.path.splitext(pdf_file_path)
            # 可以手动在本地提取,然后清洗过滤,就不需要自动提取了
            if need_extrac_img:
                parse_pdf_to_images(pdf_file_path)

            if os.path.exists(output_dir) and os.listdir(output_dir):
                # 根据文件路径生成唯一的文件ID
                file_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(pdf_file_path))
                )

                LogUtils.log_info(f"\n正在提取图片内容：{pdf_file_path}")

                file_img_documents = SimpleDirectoryReader(
                    input_dir=output_dir,
                    file_extractor=self.file_extractor,
                ).load_data(show_progress=True)

                LogUtils.log_info(
                    f"\n 从 {pdf_file_path} 中提取了 {len(file_img_documents)} 个图片文档"
                )

                # 为每个图片文档添加文件ID
                for img_document_item in file_img_documents:
                    img_document_item.metadata["file_id"] = file_id
                    img_document_item.excluded_llm_metadata_keys.append("file_id")
                    img_document_item.excluded_embed_metadata_keys.append("file_id")

                    img_document_item.metadata["image_type"] = "pdf_image"
                    img_document_item.excluded_llm_metadata_keys.append("image_type")
                    img_document_item.excluded_embed_metadata_keys.append("image_type")

                all_img_documents.extend(file_img_documents)

        LogUtils.log_info(f"总共提取了 {len(all_img_documents)} 个图片文档")
        return all_img_documents

    def load_file_list(
            self, input_file_paths: List[str],
            need_extrac_img: bool = True
    ) -> List[Document]:

        start_time = time.time()

        pdf_file_paths = [f for f in input_file_paths if f.lower().endswith(".pdf")]
        LogUtils.log_info(f"找到 {len(pdf_file_paths)} 个PDF文件")

        img_documents = (
            self._extract_pdf_image_document(
                pdf_file_paths=pdf_file_paths, need_extrac_img=need_extrac_img
            )
            if pdf_file_paths
            else []
        )

        text_documents = SimpleDirectoryReader(
            input_files=input_file_paths,
            file_extractor=self.file_extractor,
        ).load_data(show_progress=True)

        return self._process_documents(text_documents, img_documents, start_time)

    def load_file_dir(
            self, input_dir: str,
            need_extrac_img: bool = True
    ) -> list[Document]:
        if not os.path.isdir(input_dir):
            raise ValueError(f"提供的路径 '{input_dir}' 不是一个有效的目录")

        input_file_paths = [
            os.path.abspath(os.path.join(input_dir, f))
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return self.load_file_list(
            input_file_paths=input_file_paths, need_extrac_img=need_extrac_img
        )

    def _process_documents(
            self,
            text_documents: list[Document],
            img_documents: list[Document],
            start_time: float,
    ) -> list[Document]:
        """
        处理文档，添加文件ID，并计算加载时间

        Args:
            text_documents: 原始文档列表
            img_documents: 图片文档列表
            start_time: 开始时间

        Returns:
            处理后的所有文档列表
        """
        for text_document_item in text_documents:
            file_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, text_document_item.metadata["file_path"])
            )
            text_document_item.metadata["file_id"] = file_id
            text_document_item.excluded_llm_metadata_keys.append("file_id")
            text_document_item.excluded_embed_metadata_keys.append("file_id")

        elapsed_time = round(time.time() - start_time, 2)
        total_documents = len(text_documents) + len(img_documents)
        LogUtils.log_info(
            f"总共加载了 {total_documents} 个文档（包括 {len(text_documents)} 个原始文档和 {len(img_documents)} 个图片文档），耗时 {elapsed_time}秒"
        )

        return text_documents + img_documents

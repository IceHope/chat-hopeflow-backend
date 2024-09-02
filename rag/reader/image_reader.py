from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import ImageDocument

from utils.image_utils import get_image_base64_url
from utils.multi_modal_utils import get_mutil_modal_config_model


class HopeImageVisionLLMReader(BaseReader):
    """Image parser.

    Caption image using api multimodal VisionLLM eg GPT-4o

    """

    def __init__(
            self,
            parser_config: Optional[Dict] = None,
            keep_image: bool = False,
    ):
        """Init params."""
        if parser_config is None:
            pass

        self._parser_config = parser_config
        self._keep_image = keep_image

        self._lc_modul_llm = get_mutil_modal_config_model()

    def load_data(
            self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[ImageDocument]:
        """Parse file."""

        image_url = get_image_base64_url(file)

        inputs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张图片描述了什么,详细介绍下,要包含所有的专业术语",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        response = self._lc_modul_llm.invoke(inputs).content
        # print(response)

        return [
            ImageDocument(
                text=response,
                image=image_url if self._keep_image else "",
                image_path=str(file),
                metadata=extra_info or {},
            )
        ]

from typing import List

from utils.log_utils import LogUtils


class QueryManager:
    def __init__(self):
        LogUtils.log_info("QueryManager initialized")

    def query_rewrite(self, query) -> List[str]:
        # TODO: Implement query rewrite logic
        return [query]

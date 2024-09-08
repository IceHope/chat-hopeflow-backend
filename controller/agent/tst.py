idea = "《星际争霸》"

background_prompt = f"""请根据故事灵感[{idea}]创作故事的世界信息和背景故事，其中：
 世界信息需要包括世界的主要国家或地区分布，不同国家或地区的环境描写，科技水平，信仰情况等
 世界背景故事需要以时间线的形式描述世界的主要历史沿革，国家或地区之间的重大事件及带来的影响变化等
 输出格式如下
 {{
     "世界名称": "str",
     "主要国家或地区": [{{
         "名称": "str",
         "关键信息": "str",
     }}],
     "世界背景故事": ["str"],
 }}
 """

print(background_prompt)
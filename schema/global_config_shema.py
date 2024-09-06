from pydantic import BaseModel

from schema.command_shema import CommandSchema
from schema.model_schema import ModelListSchema


class GlobalConfigSchema(BaseModel):
    commands: CommandSchema
    models: ModelListSchema


if __name__ == '__main__':
    from schema.command_shema import get_command_schema
    from schema.model_schema import get_model_list_schema

    global_config = GlobalConfigSchema(
        commands=get_command_schema(),
        models=get_model_list_schema()
    )
    print(global_config)

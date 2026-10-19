from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_bge_m3_ef, generate_embeddings, get_reranker_model


class LLMProvider:

    #大语言模型
    def chat(self,model_name:str = None,json_mode:bool = False):
        """
        获取【普通文本对话】LLM 客户端
        :param model: 可选，指定模型名称，不填则使用默认配置
        :param json_mode: 是否开启 JSON 格式输出模式
        :return: 可直接调用的 LangChain LLM 客户端
        """
        return get_llm_client(model=model_name,json_mode= json_mode)
    #视觉模型
    def vision_chat(self,vision_mode_name:str =None):
        return get_llm_client(vision_mode_name or infra_config.lm_config.lv_model)
    #获取嵌入模型
    def bge_m3_embedding(self):
        return get_bge_m3_ef()
    #生成向量
    def generate_embeddings(self,texts):
        return generate_embeddings(texts)

    #重排模型
    def reranker_model(self):
        return get_reranker_model()


llm_providers = LLMProvider()
"""业务适配配置"""

# 不同业务场景的关键词和规则
BUSINESS_CONFIGS = {
    "general": {
        "keywords": [],
        "chunk_size": 512,
        "chunk_overlap": 64,
    },
    "contract": {
        "keywords": ["甲方", "乙方", "违约责任", "争议解决", "生效日期", "金额", "期限"],
        "chunk_size": 384,
        "chunk_overlap": 48,
    },
    "finance": {
        "keywords": ["利率", "费率", "账户", "余额", "账目", "税率", "百分比"],
        "chunk_size": 384,
        "chunk_overlap": 48,
    },
    "manual": {
        "keywords": ["参数", "规格", "功能", "操作", "注意事项", "故障"],
        "chunk_size": 512,
        "chunk_overlap": 64,
    },
    "standard": {
        "keywords": ["标准号", "技术要求", "试验方法", "检验规则", "公差", "精度"],
        "chunk_size": 384,
        "chunk_overlap": 48,
    },
}


def get_business_config(business_type: str = "general") -> dict:
    """获取业务配置"""
    return BUSINESS_CONFIGS.get(business_type, BUSINESS_CONFIGS["general"])

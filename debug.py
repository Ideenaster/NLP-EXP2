# 示例代码片段，可以在 infer.py 顶部或独立脚本中运行
from transformers import AutoTokenizer
# 确保这里的 DEBUG_SENTENCE 和您 infer.py 运行时实际匹配的句子完全一样
DEBUG_SENTENCE_TO_TEST = "２０１３年，成都金沙遗址博物馆与百度百科强强联手，共同打造了数字化、立体化的权威知识普及平台：成都金沙遗址博物馆之百科数字博物馆。该数字馆通过重要遗迹、藏品精粹、文化景观等分馆，全方位展示了成都金沙遗址博物馆的众多珍贵展品。"
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") # 或者您实际使用的模型名

# 测试整个句子
encoding_full = tokenizer.encode_plus(DEBUG_SENTENCE_TO_TEST, add_special_tokens=True)
print("Full sentence word_ids:", encoding_full.word_ids())

# 单独测试 "２０１３年"
test_phrase = "２０１３年"
encoding_phrase = tokenizer.encode_plus(test_phrase, add_special_tokens=False) #不加CLS,SEP
print(f"Tokens for '{test_phrase}': {tokenizer.convert_ids_to_tokens(encoding_phrase['input_ids'])}")
print(f"Word_ids for '{test_phrase}': {encoding_phrase.word_ids()}")

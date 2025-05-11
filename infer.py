import torch
import pickle
import argparse
from transformers import AutoTokenizer
from model import BertCWS # 假设 BertCWS 在 model.py 中定义
from dataloader import BERT_MODEL_NAME as DEFAULT_BERT_MODEL_NAME # 假设 BERT_MODEL_NAME 在 dataloader.py 中定义

def get_device(device_arg):
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_id2tag(datasave_path):
    with open(datasave_path, 'rb') as inp:
        _ = pickle.load(inp)  # word2id (可能不需要)
        _ = pickle.load(inp)  # id2word (可能不需要)
        _ = pickle.load(inp)  # tag2id (可能不需要)
        id2tag = pickle.load(inp)
    return id2tag

def infer(args):
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 1. 加载 Tokenizer
    print(f"Loading tokenizer: {args.bert_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    print("Tokenizer loaded.")

    # 2. 加载 id2tag
    print(f"Loading id2tag from: {args.datasave_path}...")
    id2tag = load_id2tag(args.datasave_path)
    print("id2tag loaded.")

    # 3. 加载模型
    print(f"Loading model from: {args.model_path}...")
    # BertCWS 的构造函数需要 bert_model_name 和 num_labels (即 tagset_size)
    try:
        # PyTorch 2.6+ defaults weights_only to True. Set to False if loading a full model object.
        # Ensure the model file is from a trusted source.
        loaded_data = torch.load(args.model_path, map_location=device, weights_only=False)
        if isinstance(loaded_data, BertCWS):
            model = loaded_data
            print("Loaded complete BertCWS model instance.")
        elif isinstance(loaded_data, dict): # 假设是 state_dict
            print("Loaded state_dict. Instantiating BertCWS model...")
            num_tags = len(id2tag)
            # 从 model.py 我们知道 BertCWS.__init__ 需要 bert_model_name 和 num_labels
            model = BertCWS(bert_model_name=args.bert_model_name, num_labels=num_tags)
            model.load_state_dict(loaded_data)
            print("BertCWS model instantiated and state_dict loaded.")
        else:
            raise ValueError(f"Unsupported model format in {args.model_path}. Expected BertCWS instance or state_dict.")

        model.to(device)
        model.eval()
        print("Model loaded and set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Failed to load model from {args.model_path}. Ensure the path is correct and the file contains either a BertCWS model instance or its state_dict.")
        print("If it's a state_dict, ensure BertCWS can be instantiated with --bert_model_name and num_labels derived from --datasave_path.")
        return

    # 4. 推理逻辑
    print(f"Starting inference on: {args.input_file}")
    results = []
    with open(args.input_file, 'r', encoding='utf-8') as infile, \
         open(args.output_file, 'w', encoding='utf-8') as outfile:
        for line_idx, line in enumerate(infile):
            sentence = line.strip()
            if not sentence:
                outfile.write("\n")
                continue

            # a. Tokenize
            # 参考 dataloader.py 中的 SentenceDataset 对特殊 token 和 max_len 的处理
            encoding = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True, # 通常 BERT 需要 [CLS] 和 [SEP]
                max_length=args.max_len,
                padding='max_length', # 填充到 max_len
                truncation=True,      # 截断到 max_len
                return_attention_mask=True,
                return_token_type_ids=True, # 对于单句任务，token_type_ids 通常全为0
                return_offsets_mapping=True, # 用于后处理对齐
                return_tensors='pt'   # 返回 PyTorch tensors
            )

            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            token_type_ids = encoding['token_type_ids'].to(device)
            # offset_mapping 用于后处理，它在CPU上
            # offset_mapping 的形状是 (1, seq_len, 2)，每个元素是 (char_start, char_end)
            # 对于特殊token，offset_mapping 是 (0,0)
            offset_mapping = encoding['offset_mapping'].squeeze(0).tolist()


            # b. Model infer
            with torch.no_grad():
                predicted_tag_ids = model.infer(input_ids, attention_mask, token_type_ids) # (1, seq_len)
                predicted_tag_ids = predicted_tag_ids[0] # (seq_len)

            # c. Post-processing
            # 将预测的标签 ID 序列转换为标签字符串序列
            # predicted_tag_ids 是从 CRF decode 出来的 list of ints
            predicted_tags = [id2tag[tag_id] for tag_id in predicted_tag_ids]

            # 关键：将子词级别的标签序列转换回原始词级别的分词结果
            # 我们只关心与原始文本字符对应的 token 的标签
            # BERT tokenizer 会在词前加上 ## 表示子词，或者使用 WordPiece/SentencePiece 逻辑
            # 使用 offset_mapping 来对齐

            segmented_sentence = ""
            current_char_idx = 0
            # 过滤掉特殊token [CLS], [SEP] 和 padding 对应的标签
            # 并且只处理那些对应原始文本中字符的token
            
            # 获取原始文本的字符
            original_chars = list(sentence)
            # print(f"Original: {sentence}")
            # print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())}")
            # print(f"Tags: {predicted_tags}")
            # print(f"Offsets: {offset_mapping}")

            # 简单的后处理策略：
            # 遍历原始句子的每个字符，找到它在 token 序列中对应的第一个 token
            # 使用该 token 的标签来决定分词边界
            # BIES 标签：S (单字成词), B (词的开始), I (词的中间), E (词的结束)
            
            # word_ids() 方法可以帮助将 token 映射回原始单词索引
            # encoding.word_ids() 会返回一个列表，每个 token 对应原始文本中的词（字符）索引
            # None 表示特殊 token
            word_indices = encoding.word_ids(batch_index=0) # batch_index=0 因为我们处理单个句子
            
            # print(f"Word indices: {word_indices}")

            char_tags = ['O'] * len(sentence) # 初始化每个字符的标签为 'O' (Outside)
            
            # 将 token 标签映射到字符级别
            # 对于一个字符，如果它被拆分为多个 subword token，我们通常关心第一个 subword token 的标签
            # 或者，可以采用更复杂的策略，如投票或优先考虑 B/S 标签
            
            last_word_idx = -1
            for token_idx, word_idx in enumerate(word_indices):
                if word_idx is not None: # 非特殊 token
                    # 我们只关心每个原始字符对应的第一个 token 的标签
                    if word_idx != last_word_idx: 
                        # 确保 token_idx 在 predicted_tags 的有效范围内
                        # predicted_tags 对应于 input_ids 的长度，包括特殊token
                        if token_idx < len(predicted_tags):
                            char_tags[word_idx] = predicted_tags[token_idx]
                    last_word_idx = word_idx
            
            # print(f"Char tags: {char_tags}")

            # 根据字符标签进行分词
            result_chars = []
            for i, char_token in enumerate(original_chars):
                result_chars.append(char_token)
                # 在 S 或 E 标签后加空格
                if char_tags[i] in ['S', 'E']:
                    if i < len(original_chars) -1: # 不是最后一个字符
                         result_chars.append(' ')
            
            segmented_sentence = "".join(result_chars)
            
            # 另一种更直接的基于 token 和 offset_mapping 的方法：
            # (这种方法可能更鲁棒，因为它直接使用 token 的预测)
            # segmented_output = []
            # last_char_idx = 0
            # for i in range(len(predicted_tags)): # 遍历所有 token 的预测标签
            #     start_char, end_char = offset_mapping[i]
            #     if start_char == end_char: # 特殊 token [CLS], [SEP] 或 padding
            #         continue
                
            #     token_str = sentence[start_char:end_char]
            #     tag = predicted_tags[i]
                
            #     # 我们需要将原始文本字符与 token 对应起来
            #     # 如果一个字符是某个 token 的开始，并且这个 token 的标签是 S 或 B
            #     # 或者这个字符是某个 token 的结束，并且这个 token 的标签是 S 或 E
                
            #     # 简单的策略：如果 token 的标签是 S 或 E，则在其对应的原始文本片段后加空格
            #     # 这需要小心处理 subword 的情况。
            #     # 例如 "北京大学" -> "北", "京", "大", "学"
            #     # Tokens: [CLS], 北, 京, 大, 学, [SEP]
            #     # Tags:   _,     S,  S,  B,  E,  _
            #     # Output: 北 京 大 学
                
            #     # 使用 word_ids() 的方法通常更推荐
            
            # print(f"Segmented: {segmented_sentence}")
            outfile.write(segmented_sentence + "\n")
            results.append(segmented_sentence)
            if (line_idx + 1) % 10 == 0:
                print(f"Processed {line_idx + 1} lines...")

    print(f"Inference complete. Results saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BERT-CRF Chinese Word Segmentation Inference")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file (one sentence per line).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.pkl).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for segmented results.")
    parser.add_argument("--datasave_path", type=str, required=True, help="Path to datasave.pkl for loading id2tag.")
    parser.add_argument("--bert_model_name", type=str, default=DEFAULT_BERT_MODEL_NAME, help=f"Pretrained BERT model name or path (default: {DEFAULT_BERT_MODEL_NAME}).")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length for BERT tokenizer (default: 128).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu, default: cuda if available).")

    args = parser.parse_args()
    infer(args)

"""
Example Usage:

python infer.py \
    --input_file data/test.txt \
    --model_path save/model.pkl \
    --output_file cws_bert_result.txt \
    --datasave_path data/datasave.pkl \
    --bert_model_name bert-base-chinese \
    --device cpu 
"""

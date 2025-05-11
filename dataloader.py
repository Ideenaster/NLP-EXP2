import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

# 使用 'bert-base-chinese' 作为预训练模型
BERT_MODEL_NAME = 'swtx/ernie-3.0-base-chinese'

class SentenceDataset(Dataset):
    def __init__(self, sentences, tags, tag2id, tokenizer_name=BERT_MODEL_NAME, max_len=128):
        self.sentences = sentences # 原始句子列表，每个元素是一个词列表
        self.tags = tags         # 原始标签列表，每个元素是对应句子的标签列表
        self.tag2id = tag2id
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.ignore_label_id = -100 # 用于对齐标签时忽略的ID

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx] # list of words
        tag_sequence = self.tags[idx]  # list of tags

        # BERT Tokenization
        tokens = []
        label_ids = []
        word_ids_map = [] # 用于存储每个 token 对应的原始词索引

        for word_idx, (word, tag_label) in enumerate(zip(sentence, tag_sequence)):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens: # 处理空字符串或特殊字符导致的空token列表
                word_tokens = [self.tokenizer.unk_token]

            tokens.extend(word_tokens)
            word_ids_map.extend([word_idx] * len(word_tokens)) # 每个 token 映射到原始词的索引

            # 对齐标签：只为第一个subword分配标签，其余为ignore_label_id
            if tag_label in self.tag2id:
                label_ids.append(self.tag2id[tag_label])
            else:
                # 如果标签不在 tag2id 中 (例如，数据清洗不完全)，则使用 ignore_label_id
                # 或者可以考虑用 'O' 标签的 ID
                label_ids.append(self.ignore_label_id)
                print(f"Warning: Tag '{tag_label}' not in tag2id. Using ignore_label_id for word '{word}'.")

            label_ids.extend([self.ignore_label_id] * (len(word_tokens) - 1))

        # Truncate or pad tokens, label_ids, and word_ids_map
        # 添加 [CLS] 和 [SEP]
        final_tokens = [self.tokenizer.cls_token] + tokens[:self.max_len - 2] + [self.tokenizer.sep_token]
        final_label_ids = [self.ignore_label_id] + label_ids[:self.max_len - 2] + [self.ignore_label_id] # CLS 和 SEP 的标签也忽略
        # 使用 -1 代表 [CLS], [SEP] token 的 word_id
        final_word_ids = [-1] + word_ids_map[:self.max_len - 2] + [-1]

        input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)
        attention_mask = [1] * len(input_ids)
        # token_type_ids 对于单句任务通常是全0
        token_type_ids = [0] * len(input_ids)

        # Padding
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length) # 通常 padding 的 token_type_id 也是 0
        final_label_ids = final_label_ids + ([self.ignore_label_id] * padding_length)
        # 使用 -1 代表 padding token 的 word_id
        final_word_ids = final_word_ids + ([-1] * padding_length)

        assert len(input_ids) == self.max_len, f"input_ids length: {len(input_ids)}"
        assert len(attention_mask) == self.max_len, f"attention_mask length: {len(attention_mask)}"
        assert len(token_type_ids) == self.max_len, f"token_type_ids length: {len(token_type_ids)}"
        assert len(final_label_ids) == self.max_len, f"final_label_ids length: {len(final_label_ids)}"
        assert len(final_word_ids) == self.max_len, f"final_word_ids length: {len(final_word_ids)}"

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(final_label_ids, dtype=torch.long),
            'word_ids': torch.tensor(final_word_ids, dtype=torch.long) # 传递 word_ids
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        word_ids = torch.stack([item['word_ids'] for item in batch]) # 新增

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'word_ids': word_ids # 新增
        }


if __name__ == '__main__':
    # 这是一个简化的测试，实际使用时需要加载真实的句子和标签数据
    # 假设 x_train 是 [['你好', '世界'], ['再见', '朋友']]
    # 假设 y_train 是 [['B-LOC', 'I-LOC'], ['O', 'O']]
    # 假设 tag2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, '[PAD]': -100} # [PAD] 通常用于序列标注的填充，但BERT有自己的填充和忽略机制

    # 示例数据 (需要从 datasave.pkl 中正确加载和转换)
    # 注意：原始的 x_train, y_train 是 id 化的，这里需要的是原始文本和标签字符串
    # 为了测试，我们创建一些虚拟数据
    print("Loading data for testing...")
    try:
        with open('../data/datasave.pkl', 'rb') as inp:
            word2id = pickle.load(inp)
            id2word = pickle.load(inp)
            tag2id = pickle.load(inp)
            id2tag = pickle.load(inp)
            x_train_ids = pickle.load(inp) # 这是id列表
            y_train_ids = pickle.load(inp) # 这是id列表
            # x_test_ids = pickle.load(inp)
            # y_test_ids = pickle.load(inp)
        print("Data loaded from datasave.pkl")

        # 将 id 转换回文本和标签字符串 (简化版，实际应用中需要更鲁棒的转换)
        # 这里假设 x_train_ids 和 y_train_ids 包含的是原始文本和标签，而不是id
        # 如果它们是id，你需要使用 id2word 和 id2tag 来转换它们
        # 为了演示，我们直接使用一些硬编码的文本数据
        
        # 假设我们从 datasave.pkl 加载的是原始文本和标签
        # 如果不是，需要先进行转换
        # x_train_sentences = [["中", "国", "欢", "迎", "你"], ["北", "京", "是", "首", "都"]]
        # y_train_tags = [["B-LOC", "I-LOC", "O", "O", "O"], ["B-LOC", "I-LOC", "O", "B-LOC", "I-LOC"]]
        
        # 从ID转换回文本和标签
        x_train_sentences_raw = []
        for sent_ids in x_train_ids[:5]: # 取前5个样本测试
            x_train_sentences_raw.append([id2word.get(idx, '[UNK]') for idx in sent_ids])
            
        y_train_tags_raw = []
        for tag_ids_list in y_train_ids[:5]:
            y_train_tags_raw.append([id2tag.get(idx, 'O') for idx in tag_ids_list]) # 假设 'O' 是默认标签

        print(f"Sample raw sentence: {x_train_sentences_raw[0]}")
        print(f"Sample raw tags: {y_train_tags_raw[0]}")


        # 创建新的 tag2id，确保包含 BERT Tokenizer 可能产生的特殊标记的标签（如果需要）
        # 但通常我们只对原始标签进行映射，BERT的特殊token如[CLS], [SEP], [PAD]的标签设为ignore_label_id
        # 确保 tag2id 包含所有在 y_train_tags_raw 中出现的标签
        current_tag2id = tag2id.copy() # 使用从文件中加载的tag2id
        
        # 确保 'O' 标签存在，并且为 BERT 的特殊 token 分配 ignore_label_id
        if 'O' not in current_tag2id:
             current_tag2id['O'] = max(current_tag2id.values()) + 1 if current_tag2id else 0


        train_dataset = SentenceDataset(x_train_sentences_raw, y_train_tags_raw, current_tag2id, max_len=50)
        # 注意：DataLoader 的 collate_fn 现在由 SentenceDataset.collate_fn 提供
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=SentenceDataset.collate_fn)
        
        print("\nTesting DataLoader...")
        for i, batch in enumerate(train_dataloader):
            print(f"\nBatch {i+1}:")
            print("Input IDs shape:", batch['input_ids'].shape)
            print("Attention Mask shape:", batch['attention_mask'].shape)
            print("Token Type IDs shape:", batch['token_type_ids'].shape)
            print("Labels shape:", batch['labels'].shape)
            print("Sample Input IDs:", batch['input_ids'][0][:20]) # 打印前20个token
            print("Sample Labels:", batch['labels'][0][:20])
            if i == 0: # 只测试一个batch
                break
        print("\nTest finished.")

    except FileNotFoundError:
        print("Error: ../data/datasave.pkl not found. Please ensure the data file exists.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
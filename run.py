import pickle
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report, precision_recall_fscore_support # 新增
# 假设 BERT 模型的类名为 BertCWS，并且在 model.py 中定义
from model import BertCWS # 修改为 BERT 模型类
from dataloader import SentenceDataset, BERT_MODEL_NAME # 引入新的 Dataset 和 BERT_MODEL_NAME
from torch.optim import AdamW # 从 torch.optim 导入 AdamW
from transformers.optimization import get_linear_schedule_with_warmup # get_linear_schedule_with_warmup 保持不变

def get_param():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5) # BERT 通常使用较小的学习率
    parser.add_argument('--max_epoch', type=int, default=5) # BERT 微调通常不需要太多 epoch
    parser.add_argument('--batch_size', type=int, default=16) # 根据 GPU 显存调整
    parser.add_argument('--hidden_dim', type=int, default=200) # 此参数可能不再直接用于BERT模型，或者用于BERT之上的自定义层
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--max_len', type=int, default=128, help="Max length for BERT tokenizer")
    parser.add_argument('--warmup_steps', type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay if we apply some.")
    return parser.parse_args()


def set_logger():
    log_file = os.path.join('save', 'log.txt')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m%d %H:%M:%S',
        filename=log_file,
        filemode='w',
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# 旧的 entity_split_bert 函数不再需要，将被新的评估逻辑替代

def main(args):
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(0) # 确保设备设置正确
        use_cuda = True
        device = torch.device("cuda")
    else:
        use_cuda = False
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")


    with open('data/datasave.pkl', 'rb') as inp:
        word2id = pickle.load(inp) # 可能不再直接使用
        id2word = pickle.load(inp) # 用于将id转回词，以供BERT tokenizer处理
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
        x_train_ids = pickle.load(inp) # 原始数据的ID形式
        y_train_ids = pickle.load(inp) # 原始标签的ID形式
        x_test_ids = pickle.load(inp)
        y_test_ids = pickle.load(inp)

    # 将 ID 序列转换回原始文本和标签字符串序列
    # 这是因为 BERT tokenizer 需要原始文本输入
    def convert_ids_to_raw(x_ids_list, y_ids_list, local_id2word, local_id2tag):
        raw_sentences = []
        raw_tags_sequences = []
        for sent_id_seq in x_ids_list:
            raw_sentences.append([local_id2word[idx] if 0 <= idx < len(local_id2word) else '[UNK]' for idx in sent_id_seq])
        for tag_id_seq in y_ids_list:
            raw_tags_sequences.append([local_id2tag[idx] if 0 <= idx < len(local_id2tag) else 'O' for idx in tag_id_seq]) # 假设 'O' 是未知标签的默认值
        return raw_sentences, raw_tags_sequences

    x_train_raw, y_train_raw = convert_ids_to_raw(x_train_ids, y_train_ids, id2word, id2tag)
    x_test_raw, y_test_raw = convert_ids_to_raw(x_test_ids, y_test_ids, id2word, id2tag)

    # 确保 tag2id 包含所有标签，特别是 'O'
    if 'O' not in tag2id:
        new_o_id = max(tag2id.values()) + 1 if tag2id else 0
        tag2id['O'] = new_o_id
        if new_o_id == len(id2tag):
            id2tag.append('O')
        elif new_o_id < len(id2tag):
            # This case implies 'O' was already mapped to an ID that id2tag covers,
            # but 'O' itself wasn't in tag2id. This is unusual but we'll handle assignment.
            id2tag[new_o_id] = 'O'
        else:
            # This case (new_o_id > len(id2tag)) indicates a mismatch that append won't fix alone.
            # For simplicity, we'll assume id2tag should grow. A more robust solution
            # might involve padding id2tag up to new_o_id.
            # However, given the logic, new_o_id should typically be len(id2tag).
            # If not, it implies id2tag wasn't correctly reflecting all prior max IDs.
            # We'll proceed with append, but log a warning if there's a gap.
            while len(id2tag) < new_o_id:
                id2tag.append(f'[PAD_TAG_{len(id2tag)}]') # Pad with placeholder
                logging.warning(f"Padding id2tag due to ID gap when adding 'O'. Index: {len(id2tag)-1}")
            id2tag.append('O')


    # 初始化 BERT 模型
    # 假设 BertCWS 模型定义在 model.py 中，并且接受 num_labels 参数
    model = BertCWS(bert_model_name=BERT_MODEL_NAME, num_labels=len(tag2id))
    model.to(device)

    for name, param in model.named_parameters():
        logging.debug('%s: %s, require_grad=%s' % (name, str(param.shape), str(param.requires_grad)))

    # BERT 通常使用 AdamW 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)


    train_dataset = SentenceDataset(x_train_raw, y_train_raw, tag2id, tokenizer_name=BERT_MODEL_NAME, max_len=args.max_len)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=SentenceDataset.collate_fn, # 使用新的collate_fn
        num_workers=14 # 根据系统调整
    )

    test_dataset = SentenceDataset(x_test_raw, y_test_raw, tag2id, tokenizer_name=BERT_MODEL_NAME, max_len=args.max_len)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=SentenceDataset.collate_fn,
        num_workers=14
    )
    
    t_total = len(train_dataloader) // 1 * args.max_epoch # 1是gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss = 0
        step = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # 模型需要返回损失和可能的 logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss # 假设模型输出是一个包含loss属性的对象 (HuggingFace 风格)
            
            if loss is None:
                logging.error("Loss is None. Check model's forward method.")
                continue

            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #梯度裁剪
            optimizer.step()
            scheduler.step() # 更新学习率

            step += 1
            if step % 50 == 0: # 调整日志频率
                logging.debug('Epoch %d-step %d loss: %f' % (epoch, step, loss.item()))
        
        logging.info(f"Epoch {epoch} average loss: {epoch_loss / len(train_dataloader)}")

        # --- 评估部分 ---
        model.eval()
        word_level_preds = []
        word_level_labels = []
        # test_dataset.ignore_label_id 在 dataloader 中定义为 self.ignore_label_id = -100
        # 如果 SentenceDataset 类中没有定义为类属性，需要通过实例访问
        # ignore_id = SentenceDataset.ignore_label_id # 假设 SentenceDataset 有这个类属性
        # 更安全的方式是从 dataset 实例获取
        ignore_id = test_dataset.ignore_label_id if hasattr(test_dataset, 'ignore_label_id') else -100


        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device) # (batch_size, seq_len)
                word_ids = batch['word_ids'].to(device) # (batch_size, seq_len), 新增

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2) # (batch_size, seq_len)

                for i in range(labels.shape[0]): # 遍历 batch 中的每个样本
                    seq_len = attention_mask[i].sum().item() # 实际序列长度（不包括 padding）
                    
                    current_sample_word_labels = []
                    current_sample_word_preds = []
                    
                    # 使用 word_ids 将 subword 级别的预测聚合到词级别
                    # 标签对齐策略：标签只分配给每个词的第一个 subword，其余 subword 标签为 ignore_id
                    # word_ids 将每个 token 映射到其原始词的索引，特殊 token (CLS, SEP, PAD) 的 word_id 为 -1
                    
                    for token_idx in range(seq_len): #只遍历有效 token
                        token_label = labels[i, token_idx].item()
                        token_pred = predictions[i, token_idx].item()
                        token_word_id = word_ids[i, token_idx].item()

                        if token_word_id != -1: # 是一个有效词的 token (不是 CLS, SEP, PAD)
                            if token_label != ignore_id:
                                # 这是词的第一个 subword (因为它有真实标签)
                                # 我们使用这个 subword 的预测作为整个词的预测
                                current_sample_word_labels.append(token_label)
                                current_sample_word_preds.append(token_pred)
                    
                    word_level_labels.extend(current_sample_word_labels)
                    word_level_preds.extend(current_sample_word_preds)

        if word_level_labels and word_level_preds:
            # 生成分类报告
            # 获取所有参与评估的标签ID (已排除了 ignore_id)
            report_label_ids = sorted(list(set(word_level_labels + word_level_preds)))
            
            # 从 id2tag 获取标签名称
            # 确保 id2tag 包含所有在 report_label_ids 中的 ID
            target_names = []
            valid_report_label_ids = [] # 存储实际用于报告的标签ID
            for lid in report_label_ids:
                if 0 <= lid < len(id2tag):
                    tag_name = id2tag[lid]
                    target_names.append(tag_name)
                    valid_report_label_ids.append(lid)
                else:
                    # tag_name = None # Explicitly set to None if out of bounds, though not used if not appended
                    logging.warning(f"Label ID {lid} not found in id2tag mapping (index out of bounds for list). It will be excluded from target_names in report.")

            if not valid_report_label_ids:
                logging.warning(f"Epoch {epoch} Evaluation: No valid target names found for metrics calculation after id2tag mapping.")
            else:
                # 确保 classification_report 的 labels 参数与 target_names 对应
                report = classification_report(
                    word_level_labels,
                    word_level_preds,
                    labels=valid_report_label_ids, # 明确指定报告中包含的标签类别
                    target_names=target_names,    # 这些标签类别的名称
                    digits=4,
                    zero_division=0
                )
                logging.info(f"Epoch {epoch} Evaluation Report (Word Level):\n{report}")

                # 计算并记录整体的 P, R, F1 (weighted and macro)
                # 注意: precision_recall_fscore_support 的 labels 参数也应与 classification_report 一致
                p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                    word_level_labels, word_level_preds, average='weighted', labels=valid_report_label_ids, zero_division=0
                )
                p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                    word_level_labels, word_level_preds, average='macro', labels=valid_report_label_ids, zero_division=0
                )
                logging.info(f"Epoch {epoch} Weighted - Precision: {p_w:.4f}, Recall: {r_w:.4f}, F1: {f1_w:.4f}")
                logging.info(f"Epoch {epoch} Macro    - Precision: {p_m:.4f}, Recall: {r_m:.4f}, F1: {f1_m:.4f}")
        else:
            logging.warning(f"Epoch {epoch} Evaluation: No valid word-level labels/predictions found for metrics calculation.")

        path_name = os.path.join("save", f"model_epoch{epoch}.pkl")
        # 保存整个模型或 state_dict，取决于你的偏好和后续加载方式
        # torch.save(model.state_dict(), path_name) # 更推荐保存 state_dict
        torch.save(model, path_name) # 保存整个模型，如果模型结构简单且不涉及复杂的自定义类实例
        logging.info("Model has been saved in %s" % path_name)


if __name__ == '__main__':
    set_logger()
    main(get_param())

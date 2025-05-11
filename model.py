import torch
import torch.nn as nn
from torchcrf import CRF
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence # No longer needed for BERT
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput # For standard output format


class BertCWS(nn.Module): # Renamed class

    def __init__(self, bert_model_name, num_labels, ignore_label_id=-100): # Modified __init__ signature, added ignore_label_id
        super(BertCWS, self).__init__()
        self.num_labels = num_labels # Changed from tagset_size to num_labels for clarity
        self.tagset_size = num_labels # Keep self.tagset_size if CRF or other parts use it
        self.ignore_label_id = ignore_label_id # Store ignore_label_id

        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_hidden_dim = self.bert.config.hidden_size
        
        # Linear layer to map BERT's output to tagset size
        # Dropout layer for regularization before the final classification layer
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.hidden2tag = nn.Linear(self.bert_hidden_dim, self.num_labels)

        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)

    # init_hidden is no longer needed for BERT

    def _get_bert_features(self, input_ids, attention_mask, token_type_ids=None): # Added token_type_ids
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # token_type_ids: (batch_size, seq_len)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, bert_hidden_dim)
        sequence_output = self.dropout(sequence_output) # Apply dropout
        emissions = self.hidden2tag(sequence_output)  # (batch_size, seq_len, num_labels)
        return emissions

    # Modified forward pass to use BERT features and standard HuggingFace output
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None): # Modified signature
        emissions = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        
        loss = None
        if labels is not None:
            # Prepare labels and mask for CRF
            # 1. Clone labels and replace ignore_label_id with a valid ID (e.g., 0)
            #    CRF internal computations cannot handle negative indices.
            #    The actual loss contribution is controlled by the adjusted mask.
            crf_labels = labels.clone()
            # Replace ignore_label_id with 0. Ensure 0 is a valid label ID in your scheme.
            crf_labels[labels == self.ignore_label_id] = 0

            # 2. Create a CRF-specific mask
            #    This mask considers both padding (from attention_mask)
            #    and positions where the original label was ignore_label_id.
            #    torchcrf historically preferred ByteTensor, but BoolTensor might also work.
            #    Using .byte() for broader compatibility.
            crf_specific_mask = attention_mask.byte()
            # Mask out positions where the original label was ignore_label_id
            crf_specific_mask[labels == self.ignore_label_id] = 0
            
            # Ensure the first timestep's mask is on, as required by torchcrf
            # The [CLS] token's label might be ignore_label_id, which would turn off its mask.
            # We need to ensure it's on for CRF validation.
            if crf_specific_mask.size(1) > 0: # Check if sequence length is greater than 0
                crf_specific_mask[:, 0] = 1
            
            loss = -self.crf(emissions, crf_labels, mask=crf_specific_mask, reduction='mean')
        
        return TokenClassifierOutput(
            loss=loss,
            logits=emissions
            # hidden_states=outputs.hidden_states, # Optional, if needed
            # attentions=outputs.attentions,       # Optional, if needed
        )

    # Modified infer method to use BERT features
    def infer(self, input_ids, attention_mask, token_type_ids=None): # Added token_type_ids
        emissions = self._get_bert_features(input_ids, attention_mask, token_type_ids)
        
        # Decode using CRF
        crf_mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=crf_mask)

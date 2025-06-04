from transformers import BertModel

class BertEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            input_ids=txt,
            token_type_ids=segment,
            attention_mask=mask,
            return_dict=False
        )
        return out

class BertClf(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        self.clf.apply(self.enc.bert._init_weights)  

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        logits = self.clf(x)
        return x, logits

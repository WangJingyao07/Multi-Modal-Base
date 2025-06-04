from transformers import RobertaModel

class RobertaEncoder(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

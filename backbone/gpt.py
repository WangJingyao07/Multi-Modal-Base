from transformers import GPT2Model

class GPT2Encoder(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2Model.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


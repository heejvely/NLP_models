# model fine-tuning task를 위한 사용자 함수
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in tqdm(enumerate(loader, 0)):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        
        if _%10 == 0:
            pass
            
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()

        """
        gradient를 0으로 설정
        이 작업을 하지 않을 시 학습중 backward를 해줄 때 계속 반영되어 예기치않은 방향으로 학습할 수 있음
        
        """
        loss.backward()  # 역전파 단계
        optimizer.step() # update hyper-parameters to optimizer
       
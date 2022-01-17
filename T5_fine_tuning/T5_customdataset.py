# model에 넣기 위한 형식으로 dataset 변경
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class CustomDataset:

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        # batch_encode_plus: 인코딩된 sequences 쌍의 추가 정보를 포함하여 사전의 index 반환
       
        # input_ids: 문장을 토크나이즈해서 인덱스값으로 변환
        # attention_mask: 패딩된 부분에 대해 학습에 영향을 받지 않기 위해 처리해주는 입력값
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()
        # Tensor.squeeze() => 차원이 1인 차원 제거

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
            }
            # torch.long => 64비트의 부호있는 정수(64-bit integer (signed))
            
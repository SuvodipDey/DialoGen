import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, rnn_type, n_layers, vocab_size, dropout, loss_type):
        super().__init__()
        if(rnn_type=="lstm"):
            self.rnn = nn.LSTM(input_size, input_size, n_layers, dropout = dropout)
        else:
            self.rnn = nn.GRU(input_size, input_size, n_layers, dropout = dropout)
            
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.loss_type = loss_type
        self.gelu = nn.GELU()
        
        self.v = nn.Parameter(torch.FloatTensor(input_size).uniform_(-0.1, 0.1))
        self.W_1 = nn.Linear(input_size, input_size)
        self.W_2 = nn.Linear(input_size, input_size)
        self.tanh = nn.Tanh()
        self.context_enc_fc = nn.Linear(input_size, input_size) 
        
        self.bow_head = nn.Linear(input_size, vocab_size, bias=False)
        self.loss_bow = nn.CrossEntropyLoss(ignore_index=-100)
        if(loss_type=="l1"):
            self.loss_next_utt = nn.L1Loss()
        elif(loss_type=="rmse"): 
            self.loss_next_utt = nn.MSELoss()
        else:
            self.loss_next_utt = nn.CosineSimilarity(dim=0)
    
    def get_encoding(self, turn, utt_embeds, query):
        values = utt_embeds[0:turn]
        weights = self.W_1(query) + self.W_2(values)
        f_att = torch.matmul(self.tanh(weights), self.v)
        weights = nn.functional.softmax(f_att, dim=0)
        context_enc = torch.matmul(values.T, weights)
        with torch.no_grad():
            wt = weights.tolist()    
        return context_enc, wt
    
    def forward(self, utt_list, a1, a2, doTrain, device): 
        utt_embeds = []
        for i in range(len(utt_list)): 
            utt_embeds.append(utt_list[i]['bert_enc'])
            
        utt_embeds = torch.stack(utt_embeds).to(device)
        cn, _ = self.rnn(utt_embeds)
        cn_pred = self.fc2(self.dropout(self.gelu(self.fc1(cn))))
        
        loss_dict = {'loss':0, 'bow':0, 'next_utt':0}
        loss = 0
        total_turns = len(utt_list)
        wt_list = []
        context_enc_list = []
        context_pred_list = []
        
        sys_turns = 0
        for turn in range(1, total_turns):
            
            if(turn%2==0):
                continue
            
            sys_turns+=1
            context_pred = cn_pred[turn-1]
            with torch.no_grad():
                context_target = utt_embeds[turn]
            
            context_enc, wt = self.get_encoding(turn, utt_embeds, context_pred)
            context_enc_final = self.context_enc_fc(self.dropout(self.gelu(context_enc)))  
                
            #Bag-of-words loss
            bow_ids = utt_list[turn]['bow_ids'].to(device)
            bow_logits = self.bow_head(context_enc_final)
            bow_logits = bow_logits.expand(bow_ids.size(0), -1)
            bow_loss = self.loss_bow(bow_logits, bow_ids) 
            
            #Next Utterance prediction loss
            if(self.loss_type == "l1"):
                next_utt_loss = self.loss_next_utt(context_pred, context_target)
            elif(self.loss_type == "rmse"):
                next_utt_loss = torch.sqrt(self.loss_next_utt(context_pred, context_target))
            else:
                next_utt_loss = 1.0 - self.loss_next_utt(context_pred, context_target)
               
            loss += (a1*bow_loss + a2*next_utt_loss)
            if(not doTrain):
                wt_list.append(wt)
                context_enc_list.append(context_enc)
                context_pred_list.append(context_pred)
                with torch.no_grad():
                    loss_dict['loss'] += (a1*bow_loss.item() + a2*next_utt_loss.item())
                    loss_dict['bow'] += bow_loss.item()
                    loss_dict['next_utt'] += next_utt_loss.item()
                    
        if(doTrain):
            return loss/(sys_turns)
        else:
            return loss_dict, context_enc_list, wt_list, context_pred_list